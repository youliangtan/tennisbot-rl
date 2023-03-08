import gymnasium as gym
import torch
from agent import TRPOAgent
import tennisbot
import time

def main():
    input_size = 9
    output_size = 6
    nn = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, output_size)
        )
    agent = TRPOAgent(policy=nn)

    # agent.load_model("agent.pth")
    agent.train("Tennisbot-v0", seed=0, batch_size=5000, iterations=100,
                max_episode_length=2500, verbose=True)
    agent.save_model("agent.pth")

    env = gym.make('Tennisbot-v0')
    ob = env.reset()
    while True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        # env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)


if __name__ == '__main__':
    print("start running")
    main()
