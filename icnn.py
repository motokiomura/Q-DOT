from typing import Tuple
import jax
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params, PRNGKey
from functools import partial


def update_id_pretrain(icnn: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    inputs = jnp.concatenate([batch.observations, batch.actions], -1)
    # action_dim = batch.actions.shape[1]

    def icnn_loss_fn(icnn_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pushed_inputs = jax.grad(lambda x: jnp.sum(icnn.apply({'params': icnn_params}, x)))(inputs)
        # pushed_actions = pushed_inputs[:, -action_dim:]
        # icnn_loss = ((batch.actions - pushed_actions)**2).mean()
        icnn_loss = ((inputs - pushed_inputs)**2).sum(1).mean()
        return icnn_loss, {'icnn_loss': icnn_loss}

    new_icnn, info = icnn.apply_gradient(icnn_loss_fn)

    return new_icnn, info


def update_icnn(key: PRNGKey, icnn: Model, critic: Model, value: Model, batch: Batch, double: bool,
                           loss_temp: float, obs_scale: float, actions_scale: float, push_obs: bool) -> Tuple[Model, InfoDict]:

    def icnn_loss_fn(icnn_params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        def push_inputs(icnn, obs, actions):
            obs_dim = obs.shape[1]
            scaled_obs = obs * obs_scale
            scaled_actions = actions * actions_scale
            inputs = jnp.concatenate([scaled_obs, scaled_actions], -1)
            pushed_inputs = jax.grad(lambda x: jnp.sum(icnn.apply({'params': icnn_params}, x)))(inputs)
            if push_obs:
                pushed_obs = pushed_inputs[:, :obs_dim] / obs_scale
                pushed_actions = pushed_inputs[:, obs_dim:] / actions_scale
            else:
                pushed_obs = obs
                pushed_actions = pushed_inputs[:, obs_dim:] / actions_scale
                pushed_inputs = jnp.concatenate([pushed_obs, pushed_actions], -1)
            
            return inputs, pushed_inputs, pushed_obs, pushed_actions

        inputs, pushed_inputs, pushed_obs, pushed_actions = push_inputs(icnn, batch.observations, batch.actions)


        q1, q2 = critic(pushed_obs, pushed_actions)
        if double:
            q = jnp.minimum(q1, q2)
        else:
            q = q1
            
        v = value(pushed_obs)
        
        if push_obs:
            w_dist = ((inputs - pushed_inputs)**2).sum(1)
        else:
            w_dist = ((batch.actions - pushed_actions)**2).sum(1)
        icnn_loss = loss_temp * w_dist - (q - v)
        icnn_loss = icnn_loss.mean()

        # lambda_ = 1
        # grad = grad_norm_icnn(icnn, icnn_params, inputs)
        # gp_loss = grad.mean()
        # icnn_loss += lambda_ * gp_loss
        
        return icnn_loss, {
            'icnn_loss': icnn_loss, 
            'icnn_w_dist': w_dist.mean(),
            # 'icnn_grad': grad.mean()
            }

    new_icnn, info = icnn.apply_gradient(icnn_loss_fn)

    return new_icnn, info


def grad_norm_icnn(icnn, icnn_params, inputs, lambda_=10):

    @partial(jax.vmap, in_axes=(0))
    @partial(jax.jacrev, argnums=0)
    def input_grad_fn(inputs):
        return jax.grad(lambda x: jnp.sum(icnn.apply({'params': icnn_params}, x)))(inputs)

    def grad_pen_fn(grad):
        # We use gradient penalties inspired from WGAN-LP loss which penalizes grad_norm > 1
        penalty = jnp.maximum(jnp.linalg.norm(grad, axis=-1) - 1, 0)**2
        return penalty

    grad = input_grad_fn(inputs)

    return grad_pen_fn(grad)