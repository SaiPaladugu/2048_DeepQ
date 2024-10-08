import gymnasium
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation


# Register the custom environment
register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',  # Adjust the path if necessary
    kwargs={'size': 10},
)

# Now you can use it
env = gymnasium.make('gym_examples/GridWorld-v0')
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())