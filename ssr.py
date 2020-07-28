import argparse
parser = argparse.ArgumentParser(description='Run precompute')
parser.add_argument('config', type=str, metavar='<config>', help='config json file')
parser.add_argument('--train', action="store_true", help='do training network')
parser.add_argument('--test', action="store_true", help='do testing network')
args = parser.parse_args()

import os, sys
sys.path.append('util')
from config_reader import *
config = config_reader(args.config)


if args.train:
    from model import *
    print(":: training")
    run_net(config, "train")

if args.test:
    from model import *
    print(":: testing")
    run_net(config, "test")
