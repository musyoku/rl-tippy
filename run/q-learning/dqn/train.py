from __future__ import division
from __future__ import print_function
import sys, os, chainer
import numpy as np
from datetime import datetime
from model import Model, Config
from chainer import functions, Variable
sys.path.append(os.path.join("..", "..", ".."))
from rl.playground.tippy import TippyAgent, OBSERVATION_WIDTH, OBSERVATION_HEIGHT, ACTION_NO_OP, ACTION_JUMP
from rl.utils.args import args
from rl.utils.optim import get_optimizer
from rl.utils.config import load_config, save_config
from rl.utils.print import printr, printb

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
		self.final_exploration_frame = conf.final_exploration_frame
		self.discount_factor = conf.discount_factor
		self.target_update_frequency = conf.target_update_frequency

		# replay memory
		self.replay_frames = replay_frames
		self.replay_actions = np.zeros((self.replay_memory_size,), dtype=np.uint8)
		self.replay_rewards = np.zeros((self.replay_memory_size,), dtype=np.int8)
		self.replay_episode_ends = np.zeros((self.replay_memory_size,), dtype=np.bool)

		self.replay_start_time = conf.replay_start_size
		self.ptr = 0
		self.current_episode = 1
		self.time_step_for_episode = 0
		self.total_time_step = 0
		self.policy_frozen = False
		self.avg_loss = 0
		self.last_loss = 0
		self.max_score = 0

		self.set_pipegapsize(200)

		self.train = True

		self.last_state = np.zeros((self.agent_history_length, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)

	# 行動を返す
	def agent_action(self):
		action = ACTION_NO_OP

		# 毎フレームで行動が切り替わるのは不自然なのでagent_update_frequencyごとに探索
		if self.total_time_step > self.agent_history_length and self.total_time_step % self.agent_action_frequency == 0:
			action, q_max, q_min = self.dqn.eps_greedy(self.last_state[None, :], self.exploration_rate)

		return action

	# 行動した結果を観測
	def agent_observe(self, state, action, reward, next_frame, score, remaining_lives):
		self.store_transition_in_replay_memory(state, action, reward)
		self.total_time_step += 1
		self.time_step_for_episode += 1

		self.last_state = np.roll(self.last_state, 1, axis=0)
		self.last_state[0] = state
		self.max_score = max(score, self.max_score)

		printr("episode {} - step {} - total {} - eps {:.3f} - memory size {}/{} - best score {} - loss {:.3e}".format(
			self.current_episode, self.time_step_for_episode, self.total_time_step, 
			self.exploration_rate,
			min(self.ptr, self.replay_memory_size), self.replay_memory_size, 
			self.max_score, self.last_loss))

		if self.total_time_step < self.replay_start_time:
			self.exploration_rate = self.initial_exploration_rate # ランダムに行動してreplay memoryを増やしていく
			return

		self.decrease_exploration_rate()

		if self.total_time_step > self.batchsize and self.total_time_step % 10 == 0:
			self.update_model_parameters()

		if self.total_time_step % self.target_update_frequency == 0:
			printr("")
			print("target updated.")
			self.dqn.update_target()

	# エピソード終了
	def agent_end(self, state, action, reward, score):
		self.store_transition_in_replay_memory(state, action, reward, episode_ends=True)
		self.total_time_step += 1
		self.current_episode += 1
		self.time_step_for_episode = 0

		self.last_state = np.roll(self.last_state, 1, axis=0)
		self.last_state[0] = state
		self.max_score = max(score, self.max_score)

		if self.current_episode % 100 == 0:
			self.dqn.save(os.path.join(args.sandbox, "model.hdf5"))

		if self.total_time_step < self.replay_start_time:
			return
			
		if self.total_time_step > self.batchsize and self.total_time_step % 10 == 0:
			self.update_model_parameters()

	def store_transition_in_replay_memory(self, state, action, reward, episode_ends=False):
		self.replay_frames[self.ptr] = state
		self.replay_actions[self.ptr] = action
		self.replay_rewards[self.ptr] = reward
		self.replay_episode_ends[self.ptr] = episode_ends
		self.ptr = (self.ptr + 1) % self.replay_memory_size

	def update_model_parameters(self):
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

			loss = self.compute_loss(state, action, reward, next_state, episode_ends)

			self.optimizer.update(lossfun=lambda: loss)
			self.avg_loss += float(loss.data)

			if self.total_time_step % 50 == 0:
				self.last_loss = self.avg_loss / 50
				self.avg_loss = 0

	def compute_loss(self, state, action, reward, next_state, episode_ends):
		xp = self.dqn.model.xp
		batchsize = state.shape[0]

		q = self.dqn.compute_q_value(state, train=True)
		max_target_q_data = self.dqn.compute_target_q_value(next_state).data
		max_target_q_data = xp.amax(max_target_q_data, axis=1)

		target_data = q.data.copy()

		for batch_idx in range(batchsize):
			if episode_ends[batch_idx] is True:
				new_target_value = np.sign(reward[batch_idx])
			else:
				new_target_value = np.sign(reward[batch_idx]) + self.discount_factor * max_target_q_data[batch_idx]
			action_idx = action[batch_idx]
			target_data[batch_idx, action_idx] = new_target_value

		target = Variable(target_data)
		loss = functions.clip((target - q) ** 2, 0.0, 1.0)	# clip loss
		loss = functions.sum(loss)

		# check NaN
		loss_value = float(loss.data)
		if loss_value != loss_value:
			import pdb
			pdb.set_trace()
		return loss
			
	# Exploration rate is linearly annealed to its final value
	def decrease_exploration_rate(self):
		self.exploration_rate -= 1.0 / self.final_exploration_frame
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
	dqn.load(os.path.join(args.sandbox, "model.hdf5"))
	if args.gpu_device != -1:
		dqn.to_gpu(args.gpu_device)
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
	conf = load_config(os.path.join(sandbox, "conf.json"), default=Config())
	run_training_loop()