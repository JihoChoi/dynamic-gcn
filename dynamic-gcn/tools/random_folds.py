"""
Author: Jiho Choi
- Twitter 15 / Twitter 16

References
- https://github.com/majingCUHK/Rumor_RvNN
- https://github.com/TianBian95/BiGCN
"""

import time
import random
from sklearn.model_selection import KFold
import numpy as np


# -------------------------------
#     k-fold Cross Validation
# -------------------------------


# TODO: merge with `load_labels`

def load_labels(path):  # load_resource_labels
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
def load_label_id_dict(LABEL_PATH):
    label_id_dict = {
        'true': [], 'false': [], 'unverified': [], 'non-rumor': []
    }
    for line in open(LABEL_PATH):
        elements = line.strip().split('\t')
        label, event_id = elements[0], elements[2]  # root_id
        label_id_dict[label].append(event_id)
    print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return label_id_dict


def load_id_label_dict(LABEL_PATH):
    num_labels = {'true': 0, 'false': 1, 'unverified': 2, 'non-rumor': 3}
    id_label_dict = {}  # label.txt
    for line in open(LABEL_PATH):
        elements = line.strip().split('\t')
        label, event_id = elements[0], elements[2]  # root_id
        id_label_dict[event_id] = label
    for key in id_label_dict.keys():
        id_label_dict[key] = num_labels[id_label_dict[key]]
    for num_label in num_labels.values():
        print(num_label, sum(value == num_label for value in id_label_dict.values()))
    return id_label_dict
"""


def load_k_fold_train_val_test(label_id_dict, k=5):
    k_fold_train = []  # train
    k_fold_validation = []  # validation
    k_fold_test = []  # test
    for index in range(k):
        k_fold_train.append([])
        k_fold_validation.append([])
        k_fold_test.append([])
    for label in label_id_dict:
        random.shuffle(label_id_dict[label])
        event_ids = np.array(label_id_dict[label])
        k_fold = KFold(n_splits=k)
        # train_val_folds = KFold(n_splits=k-1)  # T/V/T : 3/1/1
        train_val_folds = KFold(n_splits=k)  # T/V/T : 16/4/5
        for index, (train_validation_index, test_index) in enumerate(k_fold.split(event_ids)):
            train_validation_ids = event_ids[train_validation_index]
            for train_index, validation_index in train_val_folds.split(train_validation_ids):
                train_ids = train_validation_ids[train_index]
                validation_ids = train_validation_ids[validation_index]
                break
            test_ids = event_ids[test_index]
            k_fold_train[index].extend(train_ids)
            k_fold_validation[index].extend(validation_ids)
            k_fold_test[index].extend(test_ids)
    for index in range(k):
        random.shuffle(k_fold_train[index])
        random.shuffle(k_fold_validation[index])
        random.shuffle(k_fold_test[index])
    return k_fold_train, k_fold_validation, k_fold_test


def count_train_val_test_labels(id_label_dict, fold_x_train, fold_x_val, fold_x_test):
    train_list, val_list, test_list = [], [], []
    for id in fold_x_train:
        train_list.append(id_label_dict[id])
    for id in fold_x_val:
        val_list.append(id_label_dict[id])
    for id in fold_x_test:
        test_list.append(id_label_dict[id])
    counts = {'train': [], 'validation': [], 'test': []}
    for label_num in range(4):  # class count
        counts['train'].append(sum(value == label_num for value in train_list))
        counts['validation'].append(sum(value == label_num for value in val_list))
        counts['test'].append(sum(value == label_num for value in test_list))
    return counts


def print_folds_labels(id_label_dict, folds):
    fold_x_train, fold_x_val, fold_x_test = folds[0], folds[1], folds[2]
    for fold_index in range(5):
        counts = count_train_val_test_labels(
            id_label_dict,
            fold_x_train[fold_index],
            fold_x_val[fold_index],
            fold_x_test[fold_index]
        )
        print("Fold {}: {}".format(fold_index, counts))

"""
def print_folds_label_counts(id_label_dict, folds, k=5):
    k_fold_train, k_fold_val, k_fold_test = folds[0], folds[1], folds[2]
    for index in range(k):
        print("----------------" * 4)
        print('Fold #', index)
        print("----------------" * 4)
        train_list, val_list, test_list = [], [], []

        for label_num in range(4):
            print(label_num, end='\t')
        print()

        for id in k_fold_train[index]:  # Train Set
            train_list.append(id_label_dict[id])
        for label_num in range(4):
            print(sum(value == label_num for value in train_list), end='\t')
        print('e.g.', k_fold_train[index][0])

        for id in k_fold_val[index]:  # Validation Set
            val_list.append(id_label_dict[id])
        for label_num in range(4):
            print(sum(value == label_num for value in val_list), end='\t')
        print('e.g.', k_fold_val[index][0])

        for id in k_fold_test[index]:  # Test Set
            test_list.append(id_label_dict[id])
        for label_num in range(4):
            print(sum(value == label_num for value in test_list), end='\t')
        print('e.g.', k_fold_test[index][0])
        print("----------------" * 4, end='\n\n')
"""


def main():

    for dataset in ["Twitter16", "Twitter15"]:
        LABEL_PATH = './resources/{0}/{0}_label_all.txt'.format(dataset)
        id_label_dict, label_id_dict = load_labels(LABEL_PATH)
        folds = load_k_fold_train_val_test(label_id_dict, k=5)
        print_folds_labels(id_label_dict, folds)
        # print_folds_label_counts(id_label_dict, folds, k=5)


if __name__ == '__main__':
    start_time = time.time()  # Timer Start
    main()
    end_time = time.time()
    print("Elapsed Time: {0} seconds".format(round(end_time - start_time, 3)))
