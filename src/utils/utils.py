import argparse
import logging

from abc import ABC
from typing import Any

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--dataset", type=str, \
        default="KITTI", \
        choices=["parking", "KITTI"])
    parser.add_argument("--vo", type=str, \
        default="mono", \
        choices=["mono", "steoro"])
    return parser.parse_args()

class baseClass(ABC):
    def _extract_param(self, param_dict:dict, name:str, default:Any=None, param_type:type=None):
        res = param_dict.get(name)
        if res is None:
            return default
        else:
            if param_type is None or isinstance(res, param_type):
                return res
            else:
                logging.error('result type {} is not of type {}'.format(type(res), param_type))
                raise TypeError
