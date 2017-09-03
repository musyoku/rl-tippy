import numpy as np
from chainer import cuda

class ReplayMemory():
	def __init__(self, replay_frames_data):
		assert replay_frames_data.ndim == 4
		self.memory_size = len(replay_frames_data)
		self.agent_history_length = replay_frames_data.shape[1]
		self.ptr = 0
		self.total_num_stores = 0
		self.replay_states = replay_frames_data
		self.replay_next_states = np.zeros(replay_frames_data.shape, dtype=np.float32)
		self.replay_actions = np.zeros((self.memory_size,), dtype=np.int32)
		self.replay_rewards = np.zeros((self.memory_size,), dtype=np.float32)
		self.replay_episode_ends = np.zeros((self.memory_size,), dtype=np.bool)

	def get_current_num_stores(self):
		return min(self.total_num_stores, self.memory_size)
		
	def store_transition(self, state, action, reward, next_state, episode_ends=False):
		self.replay_states[self.ptr] = state
		self.replay_actions[self.ptr] = action
		self.replay_rewards[self.ptr] = reward
		self.replay_episode_ends[self.ptr] = episode_ends
		if episode_ends is False:
			self.replay_next_states[self.ptr] = next_state
		self.ptr = (self.ptr + 1) % self.memory_size
		self.total_num_stores += 1

	def sample_minibatch(self, batchsize, gpu_device=-1):
		randint_max = min(self.total_num_stores, self.memory_size)

		memory_indices = np.random.randint(0, randint_max, (batchsize,))
		shape_state = (batchsize, self.agent_history_length, self.replay_states.shape[1], self.replay_states.shape[2])
		shape_action = (batchsize,)

		batch_state = self.replay_states.take(memory_indices, axis=0)
		batch_next_state = self.replay_next_states.take(memory_indices, axis=0)
		batch_action = self.replay_actions.take(memory_indices)
		batch_reward = self.replay_rewards.take(memory_indices)
		batch_episode_ends = self.replay_episode_ends.take(memory_indices)

		if gpu_device >= 0:
			batch_state = cuda.to_gpu(batch_state, gpu_device)
			batch_next_state = cuda.to_gpu(batch_next_state, gpu_device)
			batch_action = cuda.to_gpu(batch_action, gpu_device)
			batch_reward = cuda.to_gpu(batch_reward, gpu_device)
			batch_episode_ends = cuda.to_gpu(batch_episode_ends, gpu_device)

		return batch_state, batch_action, batch_reward, batch_next_state, batch_episode_ends