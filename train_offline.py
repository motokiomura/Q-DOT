import os   
from typing import Tuple
import datetime
import gym
import numpy as np
import tqdm
import time
import absl
import sys
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from dataclasses import dataclass

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
import warnings
from logging_utils.logx import EpochLogger

import wandb


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('exp_name', 'dump', 'Epoch logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('temp', 1.0, 'Loss temperature for wasserstein regularization.')

flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('grad_pen', False, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1, 'Gradient penalty coefficient')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_boolean('log_loss', False, 'Use log gumbel loss')
flags.DEFINE_boolean('noise', False, 'Add noise to actions')
flags.DEFINE_float('noise_std', 0.1, 'Noise std for actions')

flags.DEFINE_float('obs_scale', 1., '')
flags.DEFINE_float('actions_scale', 1., '')
flags.DEFINE_integer('num_value_updates', 1, 'Number of value updates per iter')
flags.DEFINE_integer('num_icnn_updates', 1, 'Number of icnn updates per iter')
flags.DEFINE_float('td_coef', 1., '')

flags.DEFINE_float('icnn_lr', 3e-4, '')

flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_boolean('push_obs', False, '')
flags.DEFINE_float('expectile', None, '')
flags.DEFINE_float('awr_temp', None, 'inverse temperature for AWR')
flags.DEFINE_integer('id_epoch', 100000, '')
flags.DEFINE_float('id_thresh', 1e-3, '')

flags.DEFINE_boolean('layernorm', None, '')
flags.DEFINE_boolean('save_model', False, '')

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)



@dataclass(frozen=True)
class ConfigArgs:
    grad_pen: bool
    noise: bool
    noise_std: float
    lambda_gp: int
    max_clip: float
    num_value_updates: int
    log_loss: bool
    num_icnn_updates: int
    obs_scale: float
    actions_scale: float
    push_obs: bool


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # dataset.rewards = (dataset.rewards - 0.5) * 4
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    # save_dir = os.path.join(FLAGS.save_dir, ts_str)
    exp_id = f"results/{FLAGS.env_name}/" + FLAGS.exp_name
    log_folder = exp_id + '/'+FLAGS.exp_name+'_s'+str(FLAGS.seed) 
    save_dir = log_folder
    logger_kwargs={'output_dir':log_folder, 'exp_name':FLAGS.exp_name}

    print(FLAGS.flag_values_dict())
    os.environ["WANDB_MODE"] = "offline"
    os.environ['WANDB_DISABLED'] = 'true'
    wandb.init(project='w-dual', name=FLAGS.exp_name, config=FLAGS.flag_values_dict())

    import pandas as pd
    if os.path.isfile(log_folder+'/progress.txt'):
        try:
            df = pd.read_csv(log_folder+'/progress.txt',delim_whitespace=True)
            if  df['Iterations'].to_numpy()[-1]>900000:
                print("Exiting because already trained")
                exit()
        except:
            print("Cannot read progress.txt")
            pass
    e_logger = EpochLogger(**logger_kwargs)
    hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])

    os.makedirs(save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    if FLAGS.expectile is not None:
        kwargs['expectile'] = FLAGS.expectile
    if FLAGS.awr_temp is not None:
        kwargs['awr_temp'] = FLAGS.awr_temp
    if FLAGS.layernorm is not None:
        kwargs['layernorm'] = FLAGS.layernorm
    print(kwargs)
    args = ConfigArgs(grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      noise=FLAGS.noise,
                      max_clip=FLAGS.max_clip,
                      num_value_updates=FLAGS.num_value_updates,
                      log_loss=FLAGS.log_loss,
                      noise_std=FLAGS.noise_std,
                      num_icnn_updates=FLAGS.num_icnn_updates,
                      obs_scale=FLAGS.obs_scale,
                      actions_scale=FLAGS.actions_scale,
                      push_obs=FLAGS.push_obs,
                      )
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    loss_temp=FLAGS.temp,
                    double_q=FLAGS.double,
                    vanilla=False,
                    icnn_lr=FLAGS.icnn_lr,
                    args=args,
                    **kwargs)

    best_eval_returns = -np.inf
    eval_returns = []

    for _ in range(FLAGS.id_epoch):
        total_loss = 0.0
        num_batches = 0
        for batch in dataset.get_epoch_batches(1024):
            info = agent.icnn_id_pretrain(batch)
            total_loss += info['icnn_loss']
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        if avg_loss < FLAGS.id_thresh:
            break
        print('### loss:', avg_loss)

    for i in range(1, FLAGS.max_steps + 1):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            wandb.log(update_info, step=i)

        if i % FLAGS.eval_interval == 0:
            for k, v in update_info.items():
                print(k, ":     ",round(float(v), 5))
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            if eval_stats['return'] >= best_eval_returns:
                # Store best eval returns
                best_eval_returns = eval_stats['return']
            e_logger.log_tabular('Iterations', i)
            e_logger.log_tabular('AverageNormalizedReturn', eval_stats['return'])
            e_logger.dump_tabular()

            eval_stats[f'return_{FLAGS.env_name}'] = eval_stats['return']

            wandb.log(eval_stats, step=i)

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
    if FLAGS.save_model:
        agent.save(os.path.join(save_dir, 'model'))


    sys.exit(0)
    os._exit(0)
    raise SystemExit


if __name__ == '__main__':
    app.run(main)
