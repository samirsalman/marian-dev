#!/usr/bin/env python3

import argparse
import numpy as np
import yaml


DESC = "Prints keys and values from model.npz file."
S2S_SPECIAL_NODE = "special:model.yml"


def main():
    args = parse_args()
    model = np.load(args.model)

    if args.special:
        if S2S_SPECIAL_NODE not in model:
            print("No special Marian YAML node found in the model")
            exit(1)

        yaml_text = bytes(model[S2S_SPECIAL_NODE]).decode('ascii')
        if not args.key:
            print(yaml_text)
            exit(0)

        # fix the invalid trailing unicode character '#x0000' added to the YAML
        # string by the C++ cnpy library
        try:
            yaml_node = yaml.safe_load(yaml_text)
        except yaml.reader.ReaderError:
            yaml_node = yaml.safe_load(yaml_text[:-1])

        print(yaml_node[args.key])
    else:
        if args.key:
            if args.key not in model:
                print("Key not found")
                exit(1)
            if args.full_matrix:
                for (x, y), val in np.ndenumerate(model[args.key]):
                    print(val)
            else:
                print(model[args.key])
        else:
            total_nb_of_parameters = 0
            for key in model:
                if not key == S2S_SPECIAL_NODE:
                    total_nb_of_parameters += np.prod(model[key].shape)
                if args.matrix_info:
                    print(key, model[key].shape, model[key].dtype)
                else:
                    print(key)
            print('Total number of parameters:', total_nb_of_parameters)


def parse_args():
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("-m", "--model", help="model file", required=True)
    parser.add_argument("-k", "--key", help="print value for specific key")
    parser.add_argument("-s", "--special", action="store_true",
                        help="print values from special:model.yml node")
    parser.add_argument("-f", "--full-matrix", action="store_true",
                        help="force numpy to print full arrays for single key")
    parser.add_argument("-mi", "--matrix-info", action="store_true",
                        help="print full matrix info for all keys. Includes shape and dtype")
    return parser.parse_args()


if __name__ == "__main__":
    main()
