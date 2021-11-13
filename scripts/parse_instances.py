import sys
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-m', dest='name', default='alan', type=str)
args = parser.parse_args()

for line in sys.stdin:
    words = line.split()
    if words[0] == args.name and words[-1] != 'RUNNING':
        exit(1)
