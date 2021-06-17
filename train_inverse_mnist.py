from scripts.mnist.inverse import train

from config import inverse
import argparse

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true', help="save the model")
parser.add_argument('--folder', default='output/', type=str, help="save folder")
parser.add_argument('--name', default='default', type=str, help="the configuration for training")
parser.add_argument('--verbose', action='store_true', help="verbose")
parser.add_argument('--device', default='cpu', type=str, help="training device")
args = parser.parse_args()

# config
conf = inverse[args.name]

train(conf, args.device, args.save, args.name, args.folder, args.verbose)
