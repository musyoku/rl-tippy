import argparse
import chainer
from agent import Agent


def train():
	pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-e", type=int, default=1000)
	args = parser.parse_args()
	train()