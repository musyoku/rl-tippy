import numpy as np

class ReplayMemory():
	def __init__(self, replay_frames_data, agent_history_length=4):
		self.memory_size = len(replay_frames_data)
		self.agent_history_length = agent_history_length
		self.ptr = 0
		self.total_num_stores = 0
		self.replay_frames = replay_frames_data
		self.replay_actions = np.zeros((self.memory_size,), dtype=np.uint8)
		self.replay_rewards = np.zeros((self.memory_size,), dtype=np.float32)
		self.replay_episode_ends = np.zeros((self.memory_size,), dtype=np.bool)
		self.episode_end_indices = []

	def get_current_num_stores(self):
		return min(self.total_num_stores, self.memory_size)
		
	def store_transition(self, state, action, reward, episode_ends=False):
		self.replay_frames[self.ptr] = state
		self.replay_actions[self.ptr] = action
		self.replay_rewards[self.ptr] = reward
		self.replay_episode_ends[self.ptr] = episode_ends
		if episode_ends and self.ptr > self.agent_history_length:
			self.episode_end_indices.append(self.ptr)
		self.ptr = (self.ptr + 1) % self.memory_size
		self.total_num_stores += 1

	def sample_minibatch(self, batchsize):
		randint_max = min(self.total_num_stores, self.memory_size)

		# 普通にサンプリング
		memory_indices = np.random.randint(self.agent_history_length, randint_max, (batchsize,))
		shape_state = (batchsize, self.agent_history_length, self.replay_frames.shape[1], self.replay_frames.shape[2])
		shape_action = (batchsize,)

		state = np.empty(shape_state, dtype=np.float32)
		next_state = np.empty(shape_state, dtype=np.float32)
		action = self.replay_actions.take(memory_indices)
		reward = self.replay_rewards.take(memory_indices)
		episode_ends = self.replay_episode_ends.take(memory_indices)

		for batch_idx, memory_idx in enumerate(memory_indices):
			start = memory_idx - self.agent_history_length
			state[batch_idx] = self.replay_frames[start:memory_idx]
			next_state[batch_idx] = self.replay_frames[start + 1:memory_idx + 1]

		# 最後の一つにはエピソード終了時の状態を入れる
		# if len(self.episode_end_indices) > 0:
		# 	batch_idx = batchsize - 1
		# 	memory_idx = self.episode_end_indices[np.random.randint(0, len(self.episode_end_indices), 1)[0]]
		# 	start = memory_idx - self.agent_history_length
		# 	state[batch_idx] = self.replay_frames[start:memory_idx]
		# 	next_state[batch_idx] = self.replay_frames[start + 1:memory_idx + 1]
		# 	action[batch_idx] = self.replay_actions[memory_idx]
		# 	reward[batch_idx] = self.replay_rewards[memory_idx]

		return state, action, reward, next_state, episode_ends