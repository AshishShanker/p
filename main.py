from multi_arm_bandit import MultiArmBandit
from config import Config
from torch import manual_seed
def main(args):
    manual_seed(args.seed)
    cfg = Config(args)
    mab = MultiArmBandit(cfg)
    avgCumRegret = mab.run() # dim=(T,1)
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--iterations", type=int, default=1000)
    parser.add_argument("-t", "--horizon", type=int, default=200)
    parser.add_argument("-a", "--arms", type=int, default=3)
    parser.add_argument("-p", "--points", type=int, default=1000)
    main(parser.parse_args())