import argparse
import chainer
from model import Model
from agent import Agent


def train():
	agent = Agent()
	model = Model()
	pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-e", type=int, default=1000)
	args = parser.parse_args()
	train()