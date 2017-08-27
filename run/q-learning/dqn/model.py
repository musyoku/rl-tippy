import sys, os, copy
import numpy as np
import chainer
from chainer import Variable, cuda
sys.path.append(os.path.join("..", "..", ".."))
from rl.playground.tippy import ACTION_NO_OP, ACTION_JUMP
import rl.utils.stream as nn

class Config():
	def __init__(self):
		self.agent_history_length = 	4
		self.agent_action_frequency = 	4
		self.replay_memory_size = 		10 ** 6
		self.replay_start_size = 		5 * 10 ** 3
		self.target_update_frequency =	10 ** 3
		self.discount_factor = 			0.99
		self.initial_exploration_rate =	1.0
		self.final_exploration_rate =	0.1
		self.final_exploration_frame = 	10 ** 6
		self.no_op_max = 				30
		self.grad_clip =				1	
		self.weight_decay =				1e-6	
		self.initial_learning_rate =	0.001	
		self.lr_decay =					1	
		self.momentum =					0.9	
		self.optimizer =				"adam"	
		self.batchsize =	 			32

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
		self.target = copy.deepcopy(self.model)
		self.gpu_device = -1
		self.actions = [ACTION_NO_OP, ACTION_JUMP]
		self.no_op_count = 0
		self.no_op_max = no_op_max

	def to_gpu(self, device=0):
		cuda.get_device(device).use()
		self.model.to_gpu(device)
		self.target.to_gpu(device)
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
			with chainer.using_config("train", False):
				q_data = self.compute_q_value(state).data
				xp = self.model.xp
				action_idx = int(xp.argmax(q_data))
				q_max = float(xp.max(q_data))
				q_max = float(xp.min(q_data))

		action = self.actions[action_idx]

		# No-op
		self.no_op_count = self.no_op_count + 1 if action == ACTION_NO_OP else 0
		if self.no_op_count > self.no_op_count:
			action = self.actions[np.random.randint(1, len(self.actions))]

		return action, q_max, q_min

	def compute_q_value(self, state):
		if self.gpu_device >= 0:
			state = cuda.to_gpu(state, device=self.gpu_device)
		return self.model(state)

	def compute_target_q_value(self, state):
		with chainer.using_config("train", False):
			if self.gpu_device >= 0:
				state = cuda.to_gpu(state, device=self.gpu_device)
			return self.target(state)

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

	def update_target(self):
		self.target = copy.deepcopy(self.model)