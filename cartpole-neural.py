import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v1')  # Initialize the "Cart-Pole" environment

STREAK_TO_END = 120  # How many successful consecutive rounds are considered 'solved'
SOLVED_T = 199  # Minimal number of time steps for a round to be considered solved

class Agent:
	NUM_EPOCHS = 10000
	OBSERVATION_INPUT_DIM = [*env.observation_space.shape]
	ACTION_OUTPUT_DIM = [*env.action_space.shape]

	INITIAL_LEARNING_RATE = 0.1
	INITIAL_EXPLORATION_RATE = 1.0

	def __init__(self):
		self._session = tf.Session()
		# 'None' for arbitrary batch sizes.
		self._X = tf.placeholder(dtype=tf.float32, shape=[None] + Agent.OBSERVATION_INPUT_DIM)
		self._Y = tf.placeholder(dtype=tf.float32, shape=[None] + Agent.ACTION_OUTPUT_DIM)
		self._bIsTraining = tf.placeholder(dtype=tf.bool)  # Whether or not the model is in training mode
		self._global_step = tf.train.get_or_create_global_step()  # Used to update the learning and exploration rates

		# TODO Optimize 'decay_steps' for both!
		self._learningRate = tf.train.exponential_decay(learning_rate=Agent.INITIAL_LEARNING_RATE,
														global_step=self._global_step,
														decay_rate=0.1, decay_steps=Agent.NUM_EPOCHS//8)
		self._explorationRate = tf.train.exponential_decay(learning_rate=Agent.INITIAL_EXPLORATION_RATE,
														   global_step=self._global_step,
														   decay_rate=0.1, decay_steps=Agent.NUM_EPOCHS//8)

		self._nn_output = self._build_model(self._X, self._bIsTraining, output_dim=Agent.ACTION_OUTPUT_DIM,
											exploration_rate=self._explorationRate)

		self._loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._Y, logits=self._nn_output)
		self._train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self._loss_op, self._global_step)


	def close(self):
		self._session.close()

	def predict(self, x):
		return self._session.run(self._nn_output, feed_dict={self._X: x, self._bIsTraining: False})

	def train(self, train_data, train_labels):
		for x, y in zip(train_data, train_labels):
			feed_dict = {self._X: x, self._Y: y, self._bIsTraining: True}
			loss = self._session.run([self._train_op, self._loss_op], feed_dict=feed_dict)


	@staticmethod
	def _build_model(X, bIsTraining, output_dim, exploration_rate):
		with tf.name_scope("Model"):

			nn = tf.layers.dense(inputs=X, units=16, activation=tf.nn.relu)
			nn = tf.layers.dropout(inputs=nn, training=bIsTraining)
			nn = tf.layers.dense(inputs=nn, units=16, activation=tf.nn.relu)
			nn = tf.layers.dropout(inputs=nn, training=bIsTraining)
			nn = tf.layers.dense(inputs=nn, units=16, activation=tf.nn.relu)
			nn = tf.layers.dropout(inputs=nn, training=bIsTraining)

			nn = tf.layers.dense(inputs=nn, units=output_dim)  # TODO Maybe add activation? or softmax? try.

			# Model is done until here. 'nn' is the unscaled raw output of the model used for training with the loss op.
			# Now we consider what the model outputs.
			# If Model Is In Training Mode:
			#     Output nn
			# Else:  # Not in testing mode..
			#     If random.uniform() < exploration_rate:
			#          Output random action  # 0 or 1
			#     Else:
			#          Output actual action that the model see fits.

			logits = tf.nn.softmax(nn)  # Makes the raw outputs ('nn') to probabilities
			actual_prediction = tf.argmax(logits)  # Taking the action with the highest probability

			# Creating a boolean condition for the graph. This essentially samples a float 'x' in range [0,1) and checks
			# if: x < exploration_rate. If True, it will output a random action.
			bOutputRandomAction = tf.less(tf.random_uniform([]), exploration_rate)
			# Uniformly samples a random action from {0,1}
			# TODO min_val and max_val can be generalized with action_space.low and high, do it later.
			random_prediction = tf.random_uniform(shape=[], minval=0, maxval=output_dim, dtype=tf.int32)
			# Merging 'actual_prediction' with 'random_prediction'. Checks if x<exploration_rate and outputs accordingly.
			final_prediction_output = tf.cond(bOutputRandomAction, true_fn=lambda: random_prediction, false_fn=lambda: actual_prediction)

			# Finally: If training then outputs raw outputs, else... checks if exploring or not.
			return tf.cond(bIsTraining, true_fn=lambda: nn, false_fn=lambda: final_prediction_output)


def simulate():
	agent = Agent()
	num_streaks = 0

	episode = 0
	while True:
		if num_streaks >= STREAK_TO_END:
			break
		state_0 = env.reset()
		t = 0
		while True:
			action = agent.predict(state_0)
			observation, reward, done, info = env.step(action)

			state_0 = observation  # For next iteration
			# TODO Train the agent now. Observation is X and ____ is Y
			if done:
				print("Episode %d finished after %d time steps" % (episode, t))
				if t >= SOLVED_T:
					num_streaks += 1  # Hooray!
				else:
					num_streaks = 0  # Failed, reset.
				break
			t += 1
		episode += 1








if __name__ == '__main__':
	simulate()