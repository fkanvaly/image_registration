from scripts.mnist.inverse import train

import argparse

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true', help="save the model")
parser.add_argument('-f', '--folder', default='output/', type=str, help="save folder")
parser.add_argument('-n', '--name', default='default', type=str, help="the configuration for training")
parser.add_argument('-v' ,'--verbose', action='store_true', help="verbose")
parser.add_argument('--device', default='cpu', type=str, help="training device")

data = ["mnist", "brain"]
parser.add_argument("-d", "--data", metavar="NAME", choices=data, default="mnist",
                    help="choose a data to use, among: " + ", ".join(data))

args = parser.parse_args()

# config
if args.data=="mnist":
    from config import inverse
    conf = inverse[args.name]
else:
    from config_brain import inverse
    conf = inverse[args.name]
    
train(args.data, conf, args.device, args.save, args.name, args.folder, args.verbose)
