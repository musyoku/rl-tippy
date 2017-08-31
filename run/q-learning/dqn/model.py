import sys, os, copy
import numpy as np
import chainer
from chainer import Variable, cuda, serializers
sys.path.append(os.path.join("..", "..", ".."))
from rl.playground.tippy import ACTION_NO_OP, ACTION_JUMP
import rl.utils.stream as nn

class Config():
	def __init__(self):
		self.rl_agent_history_length = 		4
		self.rl_agent_action_interval = 	8
		self.rl_replay_memory_size = 		10 ** 5
		self.rl_replay_start_size = 		10 ** 4
		self.rl_target_update_frequency =	10 ** 4
		self.rl_eval_frequency =			10 ** 2
		self.rl_eval_num_runs = 			10
		self.rl_discount_factor = 			0.95
		self.rl_initial_exploration_rate =	0.5
		self.rl_final_exploration_rate =	0.1
		self.rl_final_exploration_frame = 	10 ** 6
		self.rl_no_op_max = 				30
		self.grad_clip =					0	
		self.weight_decay =					1e-6	
		self.initial_learning_rate =		0.001	
		self.lr_decay =						1	
		self.momentum =						0.9	
		self.optimizer =					"adam"	
		self.batchsize =	 				64
		self.clip_loss =	 				True

class Model():
	def __init__(self, no_op_max=4):
		self.model = nn.Stream(
			nn.Convolution2D(None, 32, ksize=8, stride=4),
			nn.ReLU(),
			# nn.MaxPooling2D(2),
			# nn.BatchNormalization(32),
			nn.Convolution2D(None, 64, ksize=4, stride=2),
			nn.ReLU(),
			# nn.MaxPooling2D(2),
			# nn.BatchNormalization(64),
			nn.Convolution2D(None, 64, ksize=2, stride=2),
			nn.ReLU(),
			# nn.MaxPooling2D(2),
			# nn.BatchNormalization(64),
			nn.Linear(None, 256),
			nn.ReLU(),
			nn.Linear(None, 256),
			nn.ReLU(),
			# nn.BatchNormalization(256),
			nn.Linear(None, 2),
		)
		self.update_target()
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
			q_data = self.compute_q_value(state, train=False).data
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

	def compute_q_value(self, state, train=True):
		with chainer.using_config("train", train):
			if self.gpu_device >= 0:
				state = cuda.to_gpu(state, device=self.gpu_device)
			return self.model(state)

	def compute_target_q_value(self, state):
		with chainer.using_config("train", False):
			if self.gpu_device >= 0:
				state = cuda.to_gpu(state, device=self.gpu_device)
			return self.target(state)

	def save(self, filename):
		if os.path.isfile(filename):
			os.remove(filename)
		serializers.save_hdf5(filename, self.model)

	def load(self, filename):
		if os.path.isfile(filename):
			print("loading {} ...".format(filename))
			serializers.load_hdf5(filename, self.model)

	def update_target(self):
		self.target = self.model.copy()