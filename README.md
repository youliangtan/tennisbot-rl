# tennisbot-rl

The original tennisbot model is obtained from https://github.com/CORE-Robotics-Lab/Wheelchair-Tennis-Robot

![](pybullet_env.png)

## Installation

install dependencies
 - pybullet
 - gym
 - stable-baselines3
 - simple_pid
 - sb3_contrib

Install the py pkg
```bash
cd tennisbot
pip install -e . 
```

## Run Demo

```bash
cd ..
```

Run pybullet playground
```bash
python3 playground.py --swing
```

## Train swing racket environment

Run gym training
```bash
## Add --gui to show visualization
python3 train_swing.py

# to check tensorboard
tensorboard --logdir=/tmp/ppo_swing/
```

Validate Model
```bash
python3 validate_swing.py
```

## train hit incoming tennis ball environment

Run gym training
```bash
python3 train.py
# or 
python3 train.py -s tuned_ppo
```

Validate Model
```bash
python3 validate.py
```

## Notes
 - Revert back to `gym` instead of `gymnasium` since `stable-baselines` does not support `gymnasium` yet: https://github.com/DLR-RM/stable-baselines3/pull/780
