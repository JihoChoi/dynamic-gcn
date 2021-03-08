import os
import sys
import time
import numpy as np
import json

sys.path.insert(0, './dynamic-gcn/')
from utils import print_dict
from utils import save_json_file
from utils import load_json_file
from utils import ensure_directory
from utils import print_dict

from preprocess_dataset import load_resource_labels
from preprocess_dataset import load_resource_trees


# (env) python ./dynamic-gcn/preparation/statistics.py Twitter16


def print_statistics(paths):

    _, label_id_dict = load_resource_labels(paths['resource_label'])
    trees_dict = load_resource_trees(paths['resource_tree'])


    print("label_id_dict")
    for key in label_id_dict:
        print(f"\t{key}: {len(label_id_dict[key])}")


    print(len(trees_dict))
    for key in trees_dict:
        print(trees_dict[key])
        break




def main():
    # -------------------------------
    #         PARSE ARGUMENTS
    # -------------------------------
    arg_names = ['command', 'dataset_name']
    if len(sys.argv) != 2:
        print("Please check the arguments.\n")
        print("Example usage:")
        print("python ./.../statistics.py Twitter16")
        exit()
    args = dict(zip(arg_names, sys.argv))
    dataset = args['dataset_name']

    paths = {}
    paths['resource_label'] = './resources/{0}/{0}_label_all.txt'.format(dataset)
    paths['resource_tree'] = './resources/{0}/data.TD_RvNN.vol_5000.txt'.format(dataset)

    print_statistics(paths)




if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
