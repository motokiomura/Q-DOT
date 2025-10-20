# Offline Reinforcement Learning with Wasserstein Regularization via Optimal Transport Maps

This repository contains the official implementation of the paper [Offline Reinforcement Learning with Wasserstein Regularization via Optimal Transport Maps](https://openreview.net/forum?id=RxdQBYKjtE#discussion), which introduces *Q-DOT*.


## Overview

Offline reinforcement learning (RL) aims to learn an optimal policy from a static dataset, making it particularly valuable in scenarios where data collection is costly, such as robotics. A major challenge in offline RL is distributional shift, where the learned policy deviates from the dataset distribution, potentially leading to unreliable out-of-distribution actions. To mitigate this issue, regularization techniques have been employed. While many existing methods utilize density ratio-based measures, such as the f-divergence, for regularization, we propose an approach that utilizes the Wasserstein distance, which is robust to out-of-distribution data and captures the similarity between actions. Our method employs input-convex neural networks (ICNNs) to model optimal transport maps, enabling the computation of the Wasserstein distance in a discriminator-free manner, thereby avoiding adversarial training and ensuring stable learning. Our approach demonstrates comparable or superior performance to widely used existing methods on the D4RL benchmark dataset.

## Run

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py  --temp=20
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --temp=20
```

Kitchen
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py --temp=400
```


## Citation
```
@inproceedings{
  omura2025qdot,
  title={Offline Reinforcement Learning with Wasserstein Regularization via Optimal Transport Maps},
  author={Motoki Omura and Yusuke Mukuta and Kazuki Ota and Takayuki Osa and Tatsuya Harada},
  booktitle={Reinforcement Learning Conference},
  year={2025},
  url={https://openreview.net/forum?id=RxdQBYKjtE}
}
```


## Acknowledgements
Our work builds upon the excellent implementations provided by [IQL](https://github.com/ikostrikov/implicit_q_learning) and [DVL](https://github.com/hari-sikchi/DVL), and we would like to express our sincere gratitude to them.
