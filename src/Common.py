import pickle
import argparse


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def to_pickle(path, name, data):
    with open(path + name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickle(path, name):
    with open(path + name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def print_e(text):
    print(BColors.FAIL + str("[ERROR] ") + str(text) + BColors.ENDC)


def print_w(text, title=True):
    title = "[WARNING] " if title else ""
    print(BColors.WARNING + title + str(text) + BColors.ENDC)


def print_g(text, title=True):
    title = "[INFO] " if title else ""
    print(BColors.OKGREEN + title + str(text) + BColors.ENDC)


def print_b(text, bold=False):
    if bold:
        print(BColors.BOLD + BColors.OKBLUE + str(text) + BColors.ENDC + BColors.ENDC)
    else:
        print(BColors.OKBLUE + str(text) + BColors.ENDC)

def __sort_dict_keys__(dict_cfg):
    return {k: __sort_dict_keys__(v) if isinstance(v, dict) else v for k, v in sorted(dict_cfg.items())}

def parse_cmd_args():
    """Obtener argumentos por linea de comandos"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-ct', type=str, help="City")
    parser.add_argument('-sd', type=int, help="Seed")
    parser.add_argument('-ep', type=int, help="Epoch number")
    parser.add_argument('-lr', type=float, help="Learning rate")
    parser.add_argument('-bs', type=int, help="Batch size")
    parser.add_argument('-stg', type=int, help="stage")
    parser.add_argument('-mv', type=str, help="Model version")
    parser.add_argument('-pctg', type=float, help="User percentage")
    parser.add_argument('-l2', type=float, help="L2 nu value")

    args = parser.parse_args()
    return args
