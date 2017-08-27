import sys, os
sys.path.append(os.path.join("..", "..", ".."))
from rl.playground.tippy import TippyAgent

class Agent(TippyAgent):

	def agent_step(self, action, reward, next_frame, score):
		pass

	def agent_end(self, action, reward, score):
		pass
