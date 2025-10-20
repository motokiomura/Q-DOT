"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

import policy
import value_net
from actor import update_awr_icnn as awr_icnn_update_actor
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_v_iql, update_q_iql
# from dual_critic import update_q_dual, update_v_dual

from icnn import update_id_pretrain, update_icnn
from ott.neural.networks.icnn import ICNN

from functools import partial


@jax.jit
def print_jax(x):
  jax.debug.print("jax-print: {x}", x=x)



def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=['double', 'vanilla', 'args'])
def _update_jit(
    rng: PRNGKey, icnn: Model, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, awr_temp: float, loss_temp: float, double: bool, vanilla: bool, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    for _ in range(args.num_icnn_updates):
        new_icnn, icnn_info = update_icnn(key, icnn, critic, value, batch, double=double, loss_temp=loss_temp, obs_scale=args.obs_scale, actions_scale=args.actions_scale, push_obs=args.push_obs)
        icnn = new_icnn
    for _ in range(args.num_value_updates):
        new_value, value_info = update_v_iql(target_critic, value, batch, expectile)
        new_critic, critic_info = update_q_iql(critic, new_value, batch, discount)
        value = new_value
        critic = new_critic

    new_actor, actor_info = awr_icnn_update_actor(key, new_icnn, actor, target_critic, new_value, batch, awr_temp, double, args.push_obs)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_icnn, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info,
        **icnn_info,
    }


@partial(jax.jit)
def _update_id_pretrain(
    icnn: Model, batch: Batch
) -> Tuple[PRNGKey, Model, InfoDict]:

    new_icnn, icnn_info = update_id_pretrain(icnn, batch)

    return new_icnn, icnn_info


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 icnn_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 awr_temp: float = 3.,
                 dropout_rate: Optional[float] = None,
                 layernorm: bool = False,
                 value_dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 loss_temp: float = 1.0,
                 double_q: bool = True,
                 vanilla: bool = True,
                 opt_decay_schedule: str = "cosine",
                 args=None):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.awr_temp = awr_temp
        self.loss_temp = loss_temp
        self.double_q = double_q
        self.vanilla = vanilla
        self.args = args

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, icnn_key = jax.random.split(rng, 5)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)

        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims,
                                          layer_norm=layernorm,
                                          dropout_rate=value_dropout_rate,
                                          )
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])
        
        icnn_def = ICNN(
            dim_data=(action_dim+observations.shape[-1]),
            dim_hidden=hidden_dims, 
            pos_weights=True
            )
        inputs = jnp.concatenate([observations, actions], -1)
        icnn = Model.create(
            icnn_def, 
            inputs=[icnn_key, inputs],
            tx=optax.adam(learning_rate=icnn_lr)
        )
        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.icnn = icnn
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def icnn_id_pretrain(self, batch: Batch) -> InfoDict:
        new_icnn, info = _update_id_pretrain(self.icnn, batch)
        self.icnn = new_icnn
        return info

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_icnn, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.icnn, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.awr_temp, self.loss_temp, self.double_q, self.vanilla, self.args)
        self.rng = new_rng
        self.icnn = new_icnn
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info

    def load(self, save_dir: str):
        self.actor = self.actor.load(os.path.join(save_dir, 'actor'))
        self.critic = self.critic.load(os.path.join(save_dir, 'critic'))
        self.value = self.value.load(os.path.join(save_dir, 'value'))
        self.target_critic = self.target_critic.load(os.path.join(save_dir, 'critic'))
        self.icnn = self.icnn.load(os.path.join(save_dir, 'icnn'))

    def save(self, save_dir: str):
        self.actor.save(os.path.join(save_dir, 'actor'))
        self.critic.save(os.path.join(save_dir, 'critic'))
        self.value.save(os.path.join(save_dir, 'value'))
        self.icnn.save(os.path.join(save_dir, 'icnn'))
