# coding: utf-8
from __future__ import division
from __future__ import print_function
import sys, os, chainer
import numpy as np
from datetime import datetime
from model import Model, Config
from chainer import functions, Variable
sys.path.append(os.path.join("..", "..", ".."))
from rl.playground.tippy import TippyAgent, OBSERVATION_WIDTH, OBSERVATION_HEIGHT, ACTION_NO_OP, ACTION_JUMP
from rl.utils.memory import ReplayMemory
from rl.utils.args import args
from rl.utils.optim import get_optimizer
from rl.utils.config import load_config, save_config
from rl.utils.print import printr, printb

class Agent(TippyAgent):
	def __init__(self, dqn, conf):
		super().__init__()
		self.dqn = dqn

		for key in dir(conf):
			if key.startswith("rl_"):
				setattr(self, key.replace("rl_", ""), getattr(conf, key))

		self.exploration_rate = 0

		self.current_episode = 1
		self.time_step_for_episode = 0
		self.total_time_step = 0
		self.max_score = 0

		self.last_state = np.zeros((self.agent_history_length, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)

	# 行動を返す
	def agent_action(self):
		action = ACTION_NO_OP

		if self.total_time_step % self.agent_action_interval == 0:
			action, q_max, q_min = self.dqn.eps_greedy(self.last_state[None, :], self.exploration_rate)

		return action

	# 行動した結果を観測
	def agent_observe(self, state, action, reward, next_frame, score, remaining_lives):
		self.last_state = np.roll(self.last_state, -1, axis=0)
		self.last_state[-1] = state

		self.max_score = max(score, self.max_score)
		self.total_time_step += 1

		printr("episode {} - eps {:.3f} - action {} - reward {:.3f} - best score {}".format(
			self.current_episode, self.exploration_rate, action, reward, self.max_score))

	# エピソード終了
	def agent_end(self, state, action, reward, score):
		self.total_time_step += 1
		self.time_step_for_episode = 0
		self.current_episode += 1

		self.last_state = np.roll(self.last_state, -1, axis=0)
		self.last_state[-1] = state
		self.max_score = max(score, self.max_score)

def run_greedy_loop():
	# model
	dqn = Model()
	dqn.load(os.path.join(args.sandbox, "model.hdf5"))
	if args.gpu_device != -1:
		dqn.to_gpu(args.gpu_device)

	# agent
	agent = Agent(dqn, conf)
	agent.set_stage(args.stage)
	agent.set_pipegapsize(200)
	agent.play()

if __name__ == "__main__":
	sandbox = datetime.now().strftime("sandbox-%y%m%d%H%M%S") if args.sandbox is None else args.sandbox
	try:
		os.mkdir(sandbox)
	except:
		pass
	conf = load_config(os.path.join(sandbox, "conf.json"), default=Config())
	run_greedy_loop()