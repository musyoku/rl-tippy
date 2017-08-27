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

		self.agent_history_length = conf.agent_history_length
		self.agent_update_frequency = conf.agent_update_frequency
		self.exploration_rate = conf.initial_exploration_rate

		# replay memory
		self.replay_frames = replay_frames
		self.replay_actions = np.zeros((self.replay_memory_size,), dtype=np.uint8)
		self.replay_rewards = np.zeros((self.replay_memory_size,), dtype=np.int8)
		self.replay_episode_ends = np.zeros((self.replay_memory_size,), dtype=np.bool)

		self.replay_memory_pointer = 0
		self.time_step = 0

		self.last_state = np.zeros((self.agent_history_length, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)

	def agent_action(self):
		action = ACTION_NO_OP
		if self.time_step > self.agent_history_length and self.time_step % self.agent_update_frequency == 0:
			action, q_max, q_min = self.dqn.eps_greedy(self.last_state, self.exploration_rate)
		print(action, self.exploration_rate)
		return action

	def agent_observe(self, action, reward, next_frame, score):
		self.replay_frames[self.replay_memory_pointer] = next_frame
		self.replay_actions[self.replay_memory_pointer] = action
		self.replay_rewards[self.replay_memory_pointer] = reward
		self.replay_memory_pointer = (self.replay_memory_pointer + 1) % self.replay_memory_size
		self.time_step += 1

		self.last_state = np.roll(self.last_state, 1, axis=0)
		self.last_state[0] += self.time_step

	def agent_end(self, action, reward, score):
		pass

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
	replay_frames = np.zeros((conf.replay_memory_size, 1, OBSERVATION_HEIGHT, OBSERVATION_WIDTH), dtype=np.float32)
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