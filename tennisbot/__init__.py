from gym.envs.registration import register

register(
    id="Tennisbot-v0", 
    entry_point="tennisbot.envs:TennisbotEnv",
)

register(
    id="SwingRacket-v0", 
    entry_point="tennisbot.envs:SwingRacketEnv",
)
