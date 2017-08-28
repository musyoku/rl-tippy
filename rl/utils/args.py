import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", "-g", type=int, default=0)
parser.add_argument("--sandbox", "-sandbox", type=str, default="sandbox")
parser.add_argument("--life", "-life", type=int, default=5)
args = parser.parse_args()