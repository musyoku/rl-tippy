import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", "-g", type=int, default=0)
parser.add_argument("--sandbox", type=str, default="sandbox")
parser.add_argument("--lives", type=int, default=1)
parser.add_argument("--stage", type=int, default=2)
args = parser.parse_args()