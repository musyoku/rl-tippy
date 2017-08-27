import sys, os, chainer
import numpy as np
from datetime import datetime
from model import Model
sys.path.append(os.path.join("..", "..", ".."))
from rl.playground.tippy import TippyAgent, OBSERVATION_WIDTH, OBSERVATION_HEIGHT, ACTION_NO_OP, ACTION_JUMP
from rl.utils.args import args
from rl.utils.optim import get_optimizer
from rl.utils.config import load_config, save_config

class Trainer(TippyAgent):
	def __init__(self, dqn, optimizer, replay_frames, conf):
		super().__init__()
		self.dqn = dqn
		self.optimizer = optimizer
		self.replay_memory_size = len(replay_frames)
		self.batchsize = conf.batchsize

		self.agent_history_length = conf.agent_history_length
		self.agent_action_frequency = conf.agent_action_frequency

		self.initial_exploration_rate = conf.initial_exploration_rate
		self.exploration_rate = conf.initial_exploration_rate
		self.final_exploration_rate = conf.final_exploration_rate

		# replay memory
		self.replay_frames = replay_frames
		self.replay_actions = np.zeros((self.replay_memory_size,), dtype=np.uint8)
		self.replay_rewards = np.zeros((self.replay_memory_size,), dtype=np.int8)
		self.replay_episode_ends = np.zeros((self.replay_memory_size,), dtype=np.bool)

		self.replay_start_time = conf.replay_start_size
		self.replay_memory_pointer = 0
		self.total_time_step = 0
		self.policy_frozen = False

		self.last_state = np.zeros((self.agent_history_length, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)

	# 行動を返す
	def agent_action(self):
		action = ACTION_NO_OP

		# 毎フレームで行動が切り替わるのは不自然なのでagent_update_frequencyごとに探索
		if self.total_time_step > self.agent_history_length and self.total_time_step % self.agent_action_frequency == 0:
			action, q_max, q_min = self.dqn.eps_greedy(self.last_state[None, :], self.exploration_rate)

		return action

	# 行動した結果を観測
	def agent_observe(self, action, reward, next_frame, score):
		self.replay_frames[self.replay_memory_pointer] = next_frame
		self.replay_actions[self.replay_memory_pointer] = action
		self.replay_rewards[self.replay_memory_pointer] = reward
		self.replay_memory_pointer = (self.replay_memory_pointer + 1) % self.replay_memory_size
		self.total_time_step += 1

		self.last_state = np.roll(self.last_state, 1, axis=0)
		self.last_state[0] = next_frame

		if self.total_time_step < self.replay_start_time:
			self.exploration_rate = self.initial_exploration_rate # ランダムに行動してreplay memoryを増やしていく
			return

		self.decrease_exploration_rate()

		if self.total_time_step > self.batchsize and self.total_time_step % 10 == 0:
			self.update_model()

	# エピソード終了
	def agent_end(self, action, reward, score):
		pass

	def update_model(self):
		if self.policy_frozen is False:
			# Sample random minibatch of transitions from replay memory
			randint_max = min(self.total_time_step, self.replay_memory_size) - 1
			memory_indices = np.random.randint(self.agent_history_length, randint_max, (self.batchsize,))

			shape_state = (self.batchsize, self.agent_history_length, OBSERVATION_HEIGHT, OBSERVATION_WIDTH)
			shape_action = (self.batchsize,)

			state = np.empty(shape_state, dtype=np.float32)
			next_state = np.empty(shape_state, dtype=np.float32)
			action = self.replay_actions.take(memory_indices)
			reward = self.replay_rewards.take(memory_indices)
			episode_ends = self.replay_episode_ends.take(memory_indices)

			for batch_idx, memory_idx in enumerate(memory_indices):
				start = memory_idx - self.agent_history_length
				state[batch_idx] = self.replay_frames[start:memory_idx]
				next_state[batch_idx] = self.replay_frames[start + 1:memory_idx + 1]

			
	# Exploration rate is linearly annealed to its final value
	def decrease_exploration_rate(self):
		self.exploration_rate -= 1.0 / self.final_exploration_rate
		if self.exploration_rate < self.final_exploration_rate:
			self.exploration_rate = self.final_exploration_rate

def setup_optimizer(model):
	optimizer = get_optimizer(conf.optimizer, conf.initial_learning_rate, conf.momentum)
	optimizer.setup(model)
	if conf.grad_clip > 0:
		optimizer.add_hook(chainer.optimizer.GradientClipping(conf.grad_clip))
	if conf.weight_decay > 0:
		optimizer.add_hook(chainer.optimizer.WeightDecay(conf.weight_decay))
	return optimizer

def run_training_loop():
	dqn = Model()
	optimizer = setup_optimizer(dqn.model)
	replay_frames = np.zeros((conf.replay_memory_size, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)
	agent = Trainer(dqn, optimizer, replay_frames, conf)
	agent.play()

if __name__ == "__main__":
	sandbox = datetime.now().strftime("sandbox-%y%m%d%H%M%S") if args.sandbox is None else args.sandbox
	try:
		os.mkdir(sandbox)
	except:
		pass
	conf = load_config(os.path.join(sandbox, "conf.json"))
	run_training_loop()