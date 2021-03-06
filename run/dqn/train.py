# coding: utf-8
from __future__ import division
from __future__ import print_function
import sys, os, chainer
import numpy as np
from datetime import datetime
from model import Model, Config
from chainer import functions, Variable, cuda
from rl.playground.tippy import TippyAgent, OBSERVATION_WIDTH, OBSERVATION_HEIGHT, ACTION_NO_OP, ACTION_JUMP
from rl.utils.memory import ReplayMemory
from rl.utils.args import args
from rl.utils.optim import get_optimizer
from rl.utils.config import load_config, save_config
from rl.utils.print import printr, printb

class Trainer(TippyAgent):
	def __init__(self, dqn, optimizer, memory, conf):
		super().__init__()
		self.dqn = dqn
		self.optimizer = optimizer
		self.batchsize = conf.batchsize
		self.clip_loss = conf.clip_loss
		self.memory = memory

		for key in dir(conf):
			if key.startswith("rl_"):
				setattr(self, key.replace("rl_", ""), getattr(conf, key))

		self.exploration_rate_train = self.initial_exploration_rate
		self.exploration_rate_eval = 0.05
		self.replay_start_time = self.replay_start_size

		self.current_episode = 1
		self.time_step_for_episode = 0
		self.total_time_step = 0
		self.policy_frozen = False
		self.avg_loss = 0
		self.last_loss = 0
		self.max_score = 0

		self.train = True
		self.eval_current_episode = 1

		self.reset_current_state()
		self.action_history = ACTION_NO_OP

	# 行動を返す
	def agent_action(self):
		action = ACTION_NO_OP

		if self.train:
			# 毎フレームで行動が切り替わるのは不自然なのでagent_update_frequencyごとに探索
			if self.time_step_for_episode > self.agent_history_length and self.total_time_step % self.agent_action_interval == 0:
				action, q_max, q_min = self.dqn.eps_greedy(self.current_state[None, :], self.exploration_rate_train)
				self.action_history = action
		else:
			if self.time_step_for_episode % self.agent_action_interval == 0:
				action, q_max, q_min = self.dqn.eps_greedy(self.current_state[None, :], self.exploration_rate_eval)
				self.action_history = action

		return action

	# 行動した結果を観測
	def agent_observe(self, frame, action, reward, next_frame, score, remaining_lives):
		self.udpate_current_state(frame)

		self.max_score = max(score, self.max_score)
		self.total_time_step += 1
		self.time_step_for_episode += 1

		if self.train:
			next_state = np.roll(self.current_state, -1, axis=0)
			next_state[-1] = next_frame
			self.memory.store_transition(self.current_state, action, reward, next_state)

			printr("episode {} - step {} - total {} - eps {:.3f} - action {} - reward {:.3f} - memory size {}/{} - best score {} - loss {:.3e}".format(
				self.current_episode, self.time_step_for_episode, self.total_time_step, 
				self.exploration_rate_train, action, reward,
				self.memory.get_current_num_stores(), self.replay_memory_size, 
				self.max_score, self.last_loss))

			if self.total_time_step < self.replay_start_time:
				self.exploration_rate_train = self.initial_exploration_rate # ランダムに行動してreplay memoryを増やしていく
				return

			self.decrease_exploration_rate()

			if self.total_time_step > self.batchsize and self.total_time_step % 4 == 0:
				self.update_model_parameters()

			if self.total_time_step % self.target_update_frequency == 0:
				printr("")
				print("target updated.")
				self.dqn.update_target()
		else:
			printr("eval {} - eps {:.3f} - action {} - reward {:.3f} - best score {}".format(
				self.eval_current_episode, self.exploration_rate_eval, action, reward, self.max_score))

	# エピソード終了
	def agent_end(self, frame, action, reward, score):
		self.udpate_current_state(frame)
		self.total_time_step += 1
		if self.train:
			self.memory.store_transition(self.current_state, action, reward, None, episode_ends=True)
			self.current_episode += 1

			self.max_score = max(score, self.max_score)

			if (self.current_episode - 1) % 100 == 0:
				self.dqn.save(os.path.join(args.sandbox, "model.hdf5"))

			if self.total_time_step < self.replay_start_time:
				return
				
			if (self.current_episode - 1) % self.eval_frequency == 0:
				self.toggle_eval_mode()

			if self.total_time_step > self.batchsize and self.total_time_step % 4 == 0:
				self.update_model_parameters()
		else:
			self.eval_current_episode += 1
			if self.eval_current_episode % self.eval_num_runs == 0:
				self.toggle_eval_mode()

		# リセット
		self.reset_current_state()
		self.time_step_for_episode = 0

	def udpate_current_state(self, frame):
		self.current_state = np.roll(self.current_state, -1, axis=0)
		self.current_state[-1] = frame

	def reset_current_state(self):
		self.current_state = np.zeros((self.agent_history_length, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)

	def toggle_eval_mode(self):
		if self.train:
			self.train = False
			self.eval_current_episode = 0
		else:
			self.train = True

	def update_model_parameters(self):
		if self.policy_frozen is False:
			state, action, reward, next_state, episode_ends = self.memory.sample_minibatch(self.batchsize, self.dqn.gpu_device)

			loss = self.compute_loss(state, action, reward, next_state, episode_ends)

			self.optimizer.update(lossfun=lambda: loss)
			self.avg_loss += float(loss.data)

			if self.total_time_step % 50 == 0:
				self.last_loss = self.avg_loss / 50
				self.avg_loss = 0

	def compute_loss(self, state, action, reward, next_state, episode_ends):
		batchsize = state.shape[0]
		xp = self.dqn.model.xp

		with chainer.using_config("train", True):
			q = self.dqn.compute_q_value(state)
		with chainer.no_backprop_mode():
			max_target_q_data = self.dqn.compute_target_q_value(next_state).data
			max_target_q_data = xp.amax(max_target_q_data, axis=1)

		t = reward + (1 - episode_ends) * self.discount_factor * max_target_q_data
		t = Variable(xp.reshape(t.astype(xp.float32), (-1, 1)))

		y = functions.reshape(functions.select_item(q, action), (-1, 1))

		if self.clip_loss:
			loss = functions.huber_loss(t, y, delta=1.0)
		else:
			loss = functions.mean_squared_error(t, y) / 2
		loss = functions.sum(loss)

		# check NaN
		loss_value = float(loss.data)
		if loss_value != loss_value:
			import pdb
			pdb.set_trace()
		return loss
			
	# Exploration rate is linearly annealed to its final value
	def decrease_exploration_rate(self):
		self.exploration_rate_train -= 1.0 / self.final_exploration_frame
		if self.exploration_rate_train < self.final_exploration_rate:
			self.exploration_rate_train = self.final_exploration_rate

def setup_optimizer(model):
	optimizer = get_optimizer(conf.optimizer, conf.initial_learning_rate, conf.momentum)
	optimizer.setup(model)
	if conf.grad_clip > 0:
		optimizer.add_hook(chainer.optimizer.GradientClipping(conf.grad_clip))
	if conf.weight_decay > 0:
		optimizer.add_hook(chainer.optimizer.WeightDecay(conf.weight_decay))
	return optimizer

def run_training_loop():
	# model
	dqn = Model(no_op_max=conf.rl_no_op_max)
	dqn.load(os.path.join(args.sandbox, "model.hdf5"))
	if args.gpu_device != -1:
		dqn.to_gpu(args.gpu_device)

	# optimizer
	optimizer = setup_optimizer(dqn.model)

	# replay memory
	replay_frames_data = np.zeros((conf.rl_replay_memory_size, conf.rl_agent_history_length, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)
	memory = ReplayMemory(replay_frames_data)

	# agent
	agent = Trainer(dqn, optimizer, memory, conf)
	agent.set_stage(args.stage)
	gaps = [200, 150, 100]
	agent.set_pipegapsize(gaps[args.difficulty])
	agent.play()

if __name__ == "__main__":
	sandbox = datetime.now().strftime("sandbox-%y%m%d%H%M%S") if args.sandbox is None else args.sandbox
	try:
		os.mkdir(sandbox)
	except:
		pass
	conf = load_config(os.path.join(sandbox, "conf.json"), default=Config())
	run_training_loop()