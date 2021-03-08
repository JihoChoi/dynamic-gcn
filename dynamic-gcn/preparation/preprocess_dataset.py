import sys
import os
import re
import time
import json
import random

sys.path.insert(0, './dynamic-gcn/')
from utils import print_dict
from utils import save_json_file
from utils import load_json_file
from utils import ensure_directory
from utils import path_join


def load_raw_labels(path):
    id_label_dict = {}
    label_id_dict = {
        'true': [], 'false': [], 'unverified': [], 'non-rumor': []
    }
    for line in open(path):
        label, event_id = line.strip().split(":")
        id_label_dict[event_id] = label
        label_id_dict[label].append(event_id)
    print("PATH: {0}, LEN: {1}".format(path, len(id_label_dict)))
    print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict


def load_raw_trees(path):
    pass


def load_resource_labels(path):
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


def load_resource_trees(path):
    trees_dict = {}
    for line in open(path):
        elements = line.strip().split('\t')
        event_id = elements[0]
        parent_index = elements[1]
        child_index = int(elements[2])
        word_features = elements[5]
        if event_id not in trees_dict:
            trees_dict[event_id] = {}
        trees_dict[event_id][child_index] = {
            'parent_index': parent_index,
            'word_features': word_features,
        }
    print('resource trees count:', len(trees_dict), '\n')
    return trees_dict


def load_resource_labels_weibo(path):  # Weibo Dataset
    id_label_dict = {}
    label_id_dict = {'0': [], '1': []}
    num_labels = {'0': 0, '1': 1}
    for line in open(path):
        elements = line.strip().split(' ')
        label, event_id = elements[1], elements[0]
        id_label_dict[event_id] = label
        label_id_dict[label].append(event_id)
    for key in id_label_dict.keys():
        id_label_dict[key] = num_labels[id_label_dict[key]]
    print("PATH: {0}, LEN: {1}".format(path, len(id_label_dict)))
    print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict


def load_resource_trees_weibo(path):  # Weibo
    trees_dict = {}
    for line in open(path):
        elements = line.strip().split('\t')
        event_id = elements[0]
        parent_index = elements[1]
        child_index = int(elements[2])
        word_features = elements[3]
        if event_id not in trees_dict:
            trees_dict[event_id] = {}
        trees_dict[event_id][child_index] = {
            'parent_index': parent_index,
            'word_features': word_features,
        }
    print('resource trees count:', len(trees_dict), '\n')
    return trees_dict


def raw_tree_to_timestamps(raw_tree_path, timestamps_path):  # temporal
    temporal_info = {}
    for file_name in os.listdir(raw_tree_path):
        file_id = file_name[:-4]
        if file_id not in temporal_info:
            temporal_info[file_id] = []
        for _, line in enumerate(open(raw_tree_path + file_name)):
            elem_list = re.split(r"[\'\,\->\[\]]", line.strip())
            elem_list = [x.strip() for x in elem_list if x.strip()]
            src_user_id, src_post_id, src_time = elem_list[0:3]
            dest_user_id, dest_post_id, dest_time = elem_list[3:6]
            if src_user_id == 'ROOT' and src_post_id == 'ROOT':
                _, _ = dest_user_id, dest_post_id  # root_user_id, root_post_id
            elif src_post_id != dest_post_id:  # responsive posts
                temporal_info[file_id].append(max(src_time, dest_time))
        temporal_info[file_id] = sorted(
            temporal_info[file_id], key=lambda x: float(x.strip()))
    return temporal_info


def retrieve_temporal_info(temporal_info, resource):  # trim or upsample
    # ------------------------------------------
    # Twitter: Sequential and Temporal Snapshots
    # ------------------------------------------
    resource_id_label_dict = resource['id_label_dict']
    resource_trees_dict = resource['trees_dict']
    for event_id in resource_id_label_dict:
        raw_timestamps_count = len(temporal_info[event_id])
        resource_trees_len = len(resource_trees_dict[event_id]) - 1
        if raw_timestamps_count > resource_trees_len:  # trim
            temporal_info[event_id] = temporal_info[event_id][:resource_trees_len]
        elif raw_timestamps_count < resource_trees_len:  # upsample
            diff_count = resource_trees_len - raw_timestamps_count
            if not len(temporal_info[event_id]):
                upsample = ['1.0'] * diff_count
            elif len(temporal_info[event_id]) >= diff_count:
                upsample = random.sample(temporal_info[event_id], diff_count)
            else:
                upsample = []
                for _ in range(diff_count):
                    upsample.append(random.choice(temporal_info[event_id]))
            temporal_info[event_id] += upsample
            temporal_info[event_id] = sorted(
                temporal_info[event_id], key=lambda x: float(x.strip()))
    return temporal_info


def retrieve_sequential_info_weibo(resource):
    # ---------------------------
    # Weibo: Sequential Snapshots
    # ---------------------------
    resource_id_label_dict = resource['id_label_dict']
    resource_trees_dict = resource['trees_dict']
    sequential_info = {}
    for event_id in resource_id_label_dict:
        if event_id in resource_trees_dict:
            resource_trees_len = len(resource_trees_dict[event_id]) - 1
            sequential_info[event_id] = ['1.0'] * resource_trees_len
        else:
            continue
    return sequential_info


# Load Temporal Information - Generate Sequential, Temporal Edge Index

def sequence_to_snapshot_index(temporal_info, snapshot_num):
    snapshot_edge_index = {}
    for event_id in temporal_info:
        if event_id not in snapshot_edge_index:
            snapshot_edge_index[event_id] = []
        sequence_len = len(temporal_info[event_id])
        base_edge_count = sequence_len % snapshot_num
        additional_edge_count = sequence_len // snapshot_num
        for snapshot_index in range(1, snapshot_num + 1):
            count = base_edge_count + additional_edge_count * snapshot_index
            snapshot_edge_index[event_id]
            snapshot_edge_index[event_id].append(count)
    return snapshot_edge_index


def temporal_to_snapshot_index(temporal_info, snapshot_num):
    snapshot_edge_index = {}
    for event_id in temporal_info:
        if event_id not in snapshot_edge_index:
            snapshot_edge_index[event_id] = []
        if not temporal_info[event_id]:
            snapshot_edge_index[event_id] = [0] * snapshot_num
            continue
        sequence = sorted(temporal_info[event_id],
                          key=lambda x: float(x.strip()))
        sequence = list(map(float, sequence))
        time_interval = (sequence[-1] - sequence[0]) / snapshot_num
        for snapshot_index in range(1, snapshot_num + 1):
            edge_count = 0
            for seq in sequence:
                if seq <= time_interval * snapshot_index + sequence[0]:
                    edge_count += 1
                else:
                    break
            snapshot_edge_index[event_id].append(edge_count)
        snapshot_edge_index[event_id].pop()
        snapshot_edge_index[event_id].append(len(temporal_info[event_id]))  #
    return snapshot_edge_index


def main():
    # -------------------------------
    #         PARSE ARGUMENTS
    # -------------------------------
    arg_names = ['command', 'dataset_name', 'snapshot_num']
    if len(sys.argv) != 3:
        print("Please check the arguments.\n")
        print("Example usage:")
        print("python ./dynamic-gcn/preparation/preprocess_dataset.py Twitter16 3\n")
        exit()
    args = dict(zip(arg_names, sys.argv))
    dataset, snapshot_num = args['dataset_name'], int(args['snapshot_num'])
    print_dict(args)

    paths = {}
    if dataset in ['Twitter15', 'Twitter16']:
        # --------------------------
        #         INIT PATHS
        # --------------------------
        # Input
        paths['raw'] = './data/raw/rumor_detection_acl2017/'
        paths['raw_label'] = os.path.join(paths['raw'], dataset.lower(), 'label.txt')
        paths['raw_tree'] = os.path.join(paths['raw'], dataset.lower(), 'tree/')
        paths['resource_label'] = './resources/{0}/{0}_label_all.txt'.format(dataset)
        paths['resource_tree'] = './resources/{0}/data.TD_RvNN.vol_5000.txt'.format(dataset)

        # Output (timestamp, index)
        paths['timestamps_raw'] = './data/timestamps/{}/timestamps_raw.txt'.format(dataset)
        paths['timestamps'] = './data/timestamps/{}/timestamps.txt'.format(dataset)
        paths['sequential_snapshots'] = './data/timestamps/{}/sequential_snapshots_{:02}.txt'.format(dataset, snapshot_num)
        paths['temporal_snapshots'] = './data/timestamps/{}/temporal_snapshots_{:02}.txt'.format(dataset, snapshot_num)
        print_dict(paths)

        # --------------------------------------
        #         RAW / RESOURCE DATASET
        # --------------------------------------
        raw = {
            'id_label_dict': None, 'label_id_dict': None, 'trees_dict': None,
        }
        resource = {
            'id_label_dict': None, 'label_id_dict': None, 'trees_dict': None,
        }
        raw['id_label_dict'], _ = load_raw_labels(paths['raw_label'])
        resource['id_label_dict'], _ = load_resource_labels(paths['resource_label'])
        resource['trees_dict'] = load_resource_trees(paths['resource_tree'])

        temporal_info = raw_tree_to_timestamps(paths['raw_tree'], paths['timestamps'])
        save_json_file(paths['timestamps_raw'], temporal_info)

        temporal_info = retrieve_temporal_info(temporal_info, resource)
        save_json_file(paths['timestamps'], temporal_info)
        edge_index = sequence_to_snapshot_index(temporal_info, snapshot_num)
        save_json_file(paths['sequential_snapshots'], edge_index)
        edge_index = temporal_to_snapshot_index(temporal_info, snapshot_num)
        save_json_file(paths['temporal_snapshots'], edge_index)

        print("the outputs are written at")
        print(paths['sequential_snapshots'], paths['temporal_snapshots'])


    elif dataset in ['Weibo']:
        # --------------------------
        #         INIT PATHS
        # --------------------------
        paths['resource_label'] = './resources/{0}/weibo_id_label.txt'.format(dataset)
        paths['resource_tree'] = './resources/{0}/weibotree.txt'.format(dataset)
        paths['timestamps'] = './data/timestamps/{}/timestamps.txt'.format(dataset)
        paths['sequential_snapshots'] = './data/timestamps/{}/sequential_snapshots_{:02}.txt'.format(dataset, snapshot_num)

        # --------------------------------
        #         RESOURCE DATASET
        # --------------------------------
        resource = {
            'id_label_dict': None, 'label_id_dict': None, 'trees_dict': None,
        }
        resource['id_label_dict'], _ = load_resource_labels_weibo(paths['resource_label'])
        resource['trees_dict'] = load_resource_trees_weibo(paths['resource_tree'])

        sequential_info = retrieve_sequential_info_weibo(resource)
        save_json_file(paths['timestamps'], sequential_info)
        edge_index = sequence_to_snapshot_index(sequential_info, snapshot_num)
        save_json_file(paths['sequential_snapshots'], edge_index)

    elif dataset in ['Pheme']:  # TODO:
        # --------------------------
        #         INIT PATHS
        # --------------------------
        paths['resource_label'] = './resources/{0}/pheme-label_balance.txt'.format(dataset)
        paths['resource_tree'] = './resources/{0}/pheme.vol_5000.txt'.format(dataset)
        paths['timestamps'] = './data/timestamps/{}/timestamps.txt'.format(dataset)
        paths['sequential_snapshots'] = './data/timestamps/{}/sequential_snapshots_{:02}.txt'.format(dataset, snapshot_num)

        # --------------------------------
        #         RESOURCE DATASET
        # --------------------------------
        resource = {
            'id_label_dict': None, 'label_id_dict': None, 'trees_dict': None,
        }
        resource['id_label_dict'], _ = load_resource_labels_weibo(paths['resource_label'])
        resource['trees_dict'] = load_resource_trees_weibo(paths['resource_tree'])

        sequential_info = retrieve_sequential_info_weibo(resource)
        save_json_file(paths['timestamps'], sequential_info)
        edge_index = sequence_to_snapshot_index(sequential_info, snapshot_num)
        save_json_file(paths['sequential_snapshots'], edge_index)

    else:
        print("Please check the name of a dataset.\n")
        print("E.g. Twitter15, Twitter16, Weibo")
        exit()
    


if __name__ == '__main__':
    start_time = time.time()  # Timer Start
    main()
    end_time = time.time()
    print("\nElapsed Time: {0} seconds".format(
        round(end_time - start_time, 3)))
