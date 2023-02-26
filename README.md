# tennisbot-rl

The original tennisbot model is obtained from https://github.com/CORE-Robotics-Lab/Wheelchair-Tennis-Robot

## Installation

install dependencies
 - pybullet
 - gymnasium (new version of OpenAI gym)
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
python3 main.py
```
