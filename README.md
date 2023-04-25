# tennisbot-rl

The original tennisbot model is obtained from https://github.com/CORE-Robotics-Lab/Wheelchair-Tennis-Robot

![](pybullet_env.png)

## Installation

install dependencies
 - pybullet
 - gym
 - stable-baselines3
 - simple_pid

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
python3 playground.py
```

Run gym training
```bash
python3 train.py
# or 
python3 train.py -s tuned_ppo
```

Run validate
```bash
python3 validate.py
```

## Run swing racket demo

```bash
python3 train_swing.py
```

## Notes
 - Revert back to `gym` instead of `gymnasium` since `stable-baselines` does not support `gymnasium` yet: https://github.com/DLR-RM/stable-baselines3/pull/780
