import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--dataset", type=str, \
        default="KITTI", \
        choices=["parking", "KITTI"])
    parser.add_argument("--vo", type=str, \
        default="mono", \
        choices=["mono", "steoro"])
    return parser.parse_args()