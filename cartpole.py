# Special thanks to Matthew Chan @ https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# for supplying his solution. I merely modified the code to better understand
# the basic concept of Reinforcement Learning

import gym
import numpy as np


env = gym.make('CartPole-v1')  # Initialize the "Cart-Pole" environment

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
NUM_ACTIONS = env.action_space.n  # Number of discrete actions; (left, right)

# Bounds for each discrete state
STATE_BOUNDS = np.array(list(zip(env.observation_space.low, env.observation_space.high)))
STATE_BOUNDS[3] = [-np.deg2rad(16), np.deg2rad(16)]  # Further narrowing the state bounds for theta'

BINS_BOUNDS = np.array([np.unique([0, x-1]) for x in NUM_BUCKETS])

Q_TABLE = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))  # Creating a Q-Table for each state-action pair

# Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

# Defining the simulation related constants
NUM_EPISODES = 1000  # Maximal number of episodes
# MAX_T = 250  # Maximal number of time steps for each round
STREAK_TO_END = 120  # How many successful consecutive rounds are considered 'solved'
SOLVED_T = 199  # Minimal number of time steps for a round to be considered solved
DEBUG_MODE = False


def select_action(state, explore_rate):
	if np.random.uniform() < explore_rate:  # Select a random action. Meant to explore
		return env.action_space.sample()
	else:  # Select the action with the highest q (not exploring)
		return np.argmax(Q_TABLE[state])


def get_explore_rate(t):
	return np.maximum(MIN_EXPLORE_RATE, np.minimum(1, 1.0 - np.log10((t + 1) / 25)))


def get_learning_rate(t):
	return np.maximum(MIN_LEARNING_RATE, np.minimum(0.5, 1.0 - np.log10((t + 1) / 25)))


def normalize(x, i):
	old_min, old_max = STATE_BOUNDS[i].min(), STATE_BOUNDS[i].max()
	new_min, new_max = BINS_BOUNDS[i].min(), BINS_BOUNDS[i].max()
	old_range = (old_max - old_min)
	new_range = (new_max - new_min)
	return (((x - old_min) * new_range) / old_range) + new_min


def discretize_state(state):
	indices = []
	for i in range(len(state)):
		value = state[i]
		m, M = STATE_BOUNDS[i].min(), STATE_BOUNDS[i].max()
		if value <= m:  # If value is lower than lowest bound then bin 0
			idx = 0
		elif M <= value:  # If value is greater than highest bound then last bin
			idx = NUM_BUCKETS[i] - 1
		else:
			raw_idx = normalize(value, i)
			idx = np.round(raw_idx).astype(np.int)
		indices.append(idx)
	return tuple(indices)


def simulate():
	discount_factor = 0.99  # Since the world is unchanging
	num_streaks = 0  # Number of consecutive times of success (More than 199 time steps)
	for episode in range(NUM_EPISODES):
		if num_streaks > STREAK_TO_END:  # It's considered done when it's solved over 120 times consecutively
			break
		learning_rate = get_learning_rate(episode)  # Get the learning rate w.r.t the episode
		explore_rate = get_explore_rate(episode)  # Get the exploration rate w.r.t the episode
		obv = env.reset()  # Reset the environment
		state_0 = discretize_state(obv)  # the initial state
		# for t in range(MAX_T):
		t = 0
		while True:
			# env.render()
			action = select_action(state_0, explore_rate)  # Select an action
			obv, reward, done, _ = env.step(action)  # Execute the action
			state = discretize_state(obv)  # Observe the result
			# Update the Q based on the result
			best_q = np.amax(Q_TABLE[state])
			new_q = learning_rate*(reward + discount_factor * best_q - Q_TABLE[state_0 + (action,)])
			Q_TABLE[state_0 + (action,)] += new_q
			state_0 = state  # Setting up for the next iteration
			if DEBUG_MODE:
				print("\nEpisode = %d" % episode)
				print("t = %d" % t)
				print("Action: %d" % action)
				print("State: %s" % str(state))
				print("Reward: %f" % reward)
				print("Best Q: %f" % best_q)
				print("Explore rate: %f" % explore_rate)
				print("Learning rate: %f" % learning_rate)
				print("Streaks: %d" % num_streaks)
				print("")
			if done:
				print("Episode %d finished after %d time steps" % (episode, t))
				if t >= SOLVED_T:
					num_streaks += 1  # Hooray!
				else:
					num_streaks = 0  # Failed, reset.
				break
			t += 1


if __name__ == "__main__":
	simulate()
