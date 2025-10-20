from typing import Tuple
import jax
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params, PRNGKey

import policy


def update(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float, double: bool) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': (q - v).mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_awr_icnn(key: PRNGKey, icnn: Model, actor: Model, critic: Model, value: Model,
           batch: Batch, awr_temp: float, double: bool, push_obs: bool) -> Tuple[Model, InfoDict]:
    # v = value(batch.observations)

    obs_dim = batch.observations.shape[1]  
    action_dim = batch.actions.shape[1]    
    inputs = jnp.concatenate([batch.observations, batch.actions], -1)
    pushed_inputs = jax.grad(lambda x: jnp.sum(icnn(x)))(inputs)
    if push_obs:
        pushed_obs = pushed_inputs[:, :obs_dim]
    else:
        pushed_obs = batch.observations
    pushed_actions = pushed_inputs[:, -action_dim:]

    v = value(pushed_obs)
    q1, q2 = critic(pushed_obs, pushed_actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    exp_a = jnp.exp((q - v) * awr_temp)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           pushed_obs,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(pushed_actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss, {
            'actor_loss': actor_loss, 
            'adv': (q - v).mean(),
            "action_diff_abs": (jnp.abs(batch.actions-pushed_actions)).mean(),
            "obs_diff_abs": (jnp.abs(batch.observations-pushed_obs)).mean(),
            }
    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info
