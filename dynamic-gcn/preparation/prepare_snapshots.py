"""
Author: Jiho Choi
Snapshot Preparation
    - Twitter 15 / Twitter 16
    - Weibo

References
    - https://github.com/majingCUHK/Rumor_RvNN/blob/master/model/Main_TD_RvNN.py
    - https://github.com/TianBian95/BiGCN/blob/master/Process/getTwittergraph.py
"""


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

from preprocess_dataset import load_resource_labels as load_labels

"""
def load_labels(path):
    id_label_dict = {}
    label_id_dict = {
        'true': [], 'false': [], 'unverified': [], 'non-rumor': []
    }
    num_labels = {'true': 0, 'false': 1, 'unverified': 2, 'non-rumor': 3}
    for line in open(path):
        elements = line.strip().split('\t')
        label, event_id = elements[0], elements[2]
        id_label_dict[event_id] = label
        label_id_dict[label].append(event_id)
    for key in id_label_dict.keys():
        id_label_dict[key] = num_labels[id_label_dict[key]]
    print("PATH: {0}, LEN: {1}".format(path, len(id_label_dict)))
    print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict
"""

# TODO: Validate

def load_snapshot_trees(paths, id_label_dict, sequences_dict, snapshot_num):
    trees_dict = {}
    current_snapshot = 0
    for line in open(paths['resource_tree']):  # loop for root posts
        elements = line.strip().split('\t')
        event_id = elements[0]
        parent_index, child_index = elements[1], int(elements[2])
        word_features = elements[5]
        if event_id not in id_label_dict:
            continue
        if parent_index != 'None':  # not root
            continue
        if event_id not in trees_dict:
            current_snapshot = 0
            trees_dict[event_id] = {}
            for snapshot_index in range(snapshot_num):
                trees_dict[event_id][snapshot_index] = {}
        for snapshot_index in range(current_snapshot, snapshot_num):
            trees_dict[event_id][snapshot_index][child_index] = {
                'parent_index': parent_index,
                'word_features': word_features,
            }

    print(len(trees_dict.keys()))
    print(list(trees_dict.keys())[:10])

    prev_event_id = None
    for line in open(paths['resource_tree']):
        elements = line.strip().split('\t')
        event_id, parent_index, child_index = elements[0], elements[1], int(elements[2])  # None
        _, _, word_features = int(elements[3]), int(elements[4]), elements[5]

        if prev_event_id != event_id:
            edge_index = 1  # responsive post count, without root node
            current_snapshot = 0

        prev_event_id = event_id

        if event_id not in id_label_dict:
            continue

        if parent_index == 'None':  # root post
            continue

        for snapshot_index in range(current_snapshot, snapshot_num):
            trees_dict[event_id][snapshot_index][child_index] = {
                'parent_index': parent_index,
                'word_features': word_features,
            }

        print(sequences_dict[event_id], '\t', edge_index, '\t', current_snapshot, snapshot_num, event_id)

        while current_snapshot < snapshot_num:
            if edge_index == sequences_dict[event_id][current_snapshot]:
                current_snapshot += 1
            else:
                break

        if current_snapshot == snapshot_num:  # next event_id
            continue

        if sequences_dict[event_id][current_snapshot - 1] != sequences_dict[event_id][current_snapshot]:
            edge_index += 1


    return trees_dict


class TweetNode(object):
    def __init__(self, index=None):
        self.index = index
        self.parent = None
        self.children = []
        self.word_index = []
        self.word_frequency = []


class TweetTree(object):
    def __init__(self, path, event_id, label, tree, snapshot_index, snapshot_num):
        self.graph_path = path
        self.event_id = event_id
        self.label = label
        self.tree = tree
        self.snapshot_index = snapshot_index
        self.snapshot_num = snapshot_num
        self.construct_tree()
        self.construct_matrices()  # tree -> edge_matrix
        self.construct_word_features()  # x_word_index, x_word_frequency
        self.save_local()

    @staticmethod
    def str2matrix(s):  # str = index:wordfreq index:wordfreq
        word_index, word_frequency = [], []
        for pair in s.split(' '):
            pair = pair.split(':')
            index, frequency = int(pair[0]), float(pair[1])
            if index <= 5000:
                word_index.append(index)
                word_frequency.append(frequency)
        return word_index, word_frequency

    def construct_tree(self):  # from tree_dict
        tree_dict = self.tree
        index2node = {}
        for i in tree_dict:
            index2node[i] = TweetNode(index=i)
        for j in tree_dict:
            child_index = j
            child_node = index2node[child_index]
            word_index, word_frequency = self.str2matrix(
                tree_dict[j]['word_features'])
            child_node.word_index = word_index
            child_node.word_frequency = word_frequency
            parent_index = tree_dict[j]['parent_index']
            if parent_index == 'None':  # root post
                root_index = child_index - 1
                root_word_index = child_node.word_index
                root_word_frequency = child_node.word_frequency
            else:  # responsive post
                parent_node = index2node[int(parent_index)]
                child_node.parent = parent_node
                parent_node.children.append(child_node)
        root_features = np.zeros([1, 5000])
        if len(root_word_index) > 0:
            root_features[0, np.array(root_word_index)] = np.array(root_word_frequency)
        self.index2node = index2node
        self.root_index = root_index
        self.root_features = root_features

    def construct_matrices(self):  # tree2matrix, adjacency
        index2node = self.index2node
        row = []
        col = []
        x_word_index_list = []
        x_word_frequency_list = []
        for index_i in sorted(list(index2node.keys())):
            child_index = []
            for child_node in index2node[index_i].children:
                child_index.append(child_node.index)
            for index_j in sorted(child_index):
                row.append(index_i-1)
                col.append(index_j-1)
        edge_matrix = [row, col]  # TODO: shift
        # shift indices
        # - new adjacency matrix for PyTorch Geometric
        index_map = {}
        shifted_index = 0
        for i in sorted(set(row).union(set(col))):
            index_map[i] = shifted_index
            shifted_index += 1
            x_word_index_list.append(index2node[i+1].word_index)
            x_word_frequency_list.append(index2node[i+1].word_frequency)
        new_row = []
        new_col = []
        for row_elem in row:
            new_row.append(index_map[row_elem])
        for col_elem in col:
            new_col.append(index_map[col_elem])
        edge_matrix = [new_row, new_col]  # TODO: shift

        self.root_index = index_map[self.root_index]
        self.edge_matrix = edge_matrix
        self.x_word_index_list = x_word_index_list
        self.x_word_frequency_list = x_word_frequency_list

    def construct_word_features(self):
        x_word_index_list = self.x_word_index_list
        x_word_frequency_list = self.x_word_frequency_list
        x_x = np.zeros([len(x_word_index_list), 5000])
        for i in range(len(x_word_index_list)):
            if len(x_word_index_list[i]) > 0:
                x_x[i, np.array(x_word_index_list[i])] = np.array(x_word_frequency_list[i])
        self.x_x = x_x

    def save_local(self):
        root_index = np.array(self.root_index)
        root_features = np.array(self.root_features)
        edge_index = np.array(self.edge_matrix)
        x_x = np.array(self.x_x)
        label = np.array(self.label)
        FILE_PATH = "{}/{}_{}_{}.npz".format(
            self.graph_path, self.event_id, self.snapshot_index, self.snapshot_num
        )
        np.savez(  # save snapshots
            FILE_PATH,
            x=x_x, y=label, edge_index=edge_index,
            root_index=root_index, root=root_features
        )


def main():
    # -------------------------------
    #         PARSE ARGUMENTS
    # -------------------------------
    arg_names = ['command', 'dataset_name', 'dataset_type', 'snapshot_num']
    if len(sys.argv) != 4:
        print("Please check the arguments.\n")
        print("Example usage:")
        print("python ./.../prepare_snapshots.py Twitter16 sequential 3")
        exit()
    args = dict(zip(arg_names, sys.argv))
    dataset = args['dataset_name']
    dataset_type = args['dataset_type']
    snapshot_num = int(args['snapshot_num'])
    print_dict(args)

    # --------------------------
    #         INIT PATHS
    # --------------------------
    paths = {}
    if dataset in ['Twitter15', 'Twitter16']:
        paths['resource_label'] = './resources/{0}/{0}_label_all.txt'.format(dataset)
        paths['resource_tree'] = './resources/{0}/data.TD_RvNN.vol_5000.txt'.format(dataset)
        paths['timestamps'] = './data/timestamps/{}/timestamps.txt'.format(dataset)
        paths['snapshot_index'] = './data/timestamps/{}/{}_snapshots_{:02}.txt'.format(dataset, dataset_type, snapshot_num)
        paths['graph'] = './data/graph/{0}/{1}_snapshot/'.format(dataset, dataset_type)
    elif dataset in ['Weibo']:
        exit()
    else:
        exit()

    # ----------------------------------
    #         GENERATE SNAPSHOTS
    # ----------------------------------

    id_label_dict, _ = load_labels(paths['resource_label'])
    sequences_dict = load_json_file(paths['snapshot_index'])
    trees_dict = load_snapshot_trees(paths, id_label_dict, sequences_dict, snapshot_num)

    ensure_directory(paths['graph'])
    for index, event_id in enumerate(id_label_dict.keys()):
        # print("[{:04d}/{:04d}]".format(index, len(id_label_dict.keys()) - 1))
        if len(trees_dict[event_id][0]) < 2:  # no responsive post
            print("no responsive post", event_id, len(trees_dict[event_id][0]))
            continue

        for snapshot_index in range(snapshot_num):
            TweetTree(
                paths['graph'],
                event_id,
                id_label_dict[event_id],
                trees_dict[event_id][snapshot_index],
                snapshot_index,
                snapshot_num,
            )


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
