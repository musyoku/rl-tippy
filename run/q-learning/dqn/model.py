import sys, os
import numpy as np
from chainer import Variable, cuda
sys.path.append(os.path.join("..", "..", ".."))
from rl.playground.tippy import ACTION_NO_OP, ACTION_JUMP
import rl.utils.stream as nn

class Model():
	def __init__(self, no_op_max=4):
		self.model = nn.Stream(
			nn.Convolution2D(None, 16, ksize=8),
			nn.BatchNormalization(16),
			nn.ReLU(),
			nn.Convolution2D(None, 32, ksize=4),
			nn.BatchNormalization(32),
			nn.ReLU(),
			nn.Linear(None, 256),
			nn.BatchNormalization(256),
			nn.ReLU(),
			nn.Linear(None, 2),
		)
		self.gpu_device = -1
		self.actions = [ACTION_NO_OP, ACTION_JUMP]
		self.no_op_count = 0
		self.no_op_max = no_op_max

	def to_gpu(self, device=0):
		cuda.get_device(device).use()
		self.model.to_gpu(device)
		self.gpu_device = device

	def eps_greedy(self, state, exploration_rate):
		prop = np.random.uniform()
		q_max = None
		q_min = None
		if prop < exploration_rate:
			# select a random action
			action_idx = np.random.randint(0, len(self.actions))
		else:
			# select a greedy action
			with chianer.using_config("train", False):
				state = Variable(state)
				if self.gpu_device >= 0:
					state.to_gpu(self.gpu_device)
				q_data = self.model(state).data
				xp = self.model.xp
				action_idx = xp.argmax(q_data)
				q_max = xp.max(q_data)
				q_max = xp.min(q_data)

		action = self.actions(action_idx)

		# No-op
		self.no_op_count = self.no_op_count + 1 if action == ACTION_NO_OP else 0
		if self.no_op_count > self.no_op_count:
			action = self.actions[np.random.randint(1, len(self.actions))]

		return action, q_max, q_min

	def save(self):
		filename = "model.hdf5"
		if os.path.isfile(filename):
			os.remove(filename)
		serializers.save_hdf5(filename, self.model)

	def load(self):
		filename = "/model.hdf5"
		if os.path.isfile(filename):
			print("loading {} ...".format(filename))
			serializers.load_hdf5(filename, self.model)
