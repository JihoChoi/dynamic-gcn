# Calculate accuracy, precision, recall, and F1 score
# Author: Jiho Choi
#
# Reference:
#   https://github.com/majingCUHK/Rumor_RvNN/blob/master/model/evaluate.py

from operator import add

def evaluation(prediction, y):  # 4-class: T, F, U, N
    TP = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    FP = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    FN = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    TN = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    for i in range(len(y)):
        pred, real = prediction[i], y[i]
        for label in range(4):
            if label == pred and label == real:
                TP[label] += 1
            if label == pred and label != real:
                FP[label] += 1
            if label != pred and label == real:
                FN[label] += 1
            if label != pred and label != real:
                TN[label] += 1

    acc_all = round(sum(TP.values()) / float(len(y)), 4)
    accuracy = {0: None, 1: None, 2: None, 3: None}
    precision = {0: None, 1: None, 2: None, 3: None}
    recall = {0: None, 1: None, 2: None, 3: None}
    F1 = {0: None, 1: None, 2: None, 3: None}

    for l in range(4):
        label_all = (TP[l] + FP[l] + FN[l] + TN[l])
        accuracy[l] = round((TP[l] + TN[l]) / label_all, 4)

        if (TP[l] + FP[l]) == 0:
            precision[l] = 0
        else:
            precision[l] = round(TP[l] / (TP[l] + FP[l]), 4)

        relavant = (TP[l] + FN[l])
        recall[l] = round(TP[l] / relavant, 4) if relavant else 0

        PR = (precision[l] + recall[l])
        F1[l] = round(2 * precision[l] * recall[l] / PR, 4) if PR else 0

    results = {
        'acc_all': acc_all,
        'C0': {'acc': accuracy[0], 'prec': precision[0], 'rec': recall[0], 'F1': F1[0]},
        'C1': {'acc': accuracy[1], 'prec': precision[1], 'rec': recall[1], 'F1': F1[1]},
        'C2': {'acc': accuracy[2], 'prec': precision[2], 'rec': recall[2], 'F1': F1[2]},
        'C3': {'acc': accuracy[3], 'prec': precision[3], 'rec': recall[3], 'F1': F1[3]},
    }
    return results


def merge_batch_eval_list(batch_eval_results):
    eval_results = {}
    batch_num = len(batch_eval_results)

    # Initialize
    # for key in batch_eval_results[0].keys():
    #     if key not in eval_results:
    #         if not isinstance(batch_eval_results[0][key], dict):
    #             eval_results[key] = 0.0
    #         else:
    #             eval_results[key] = {
    #                 'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'F1': 0.0
    #             }
    eval_results = {
        'acc_all': 0.0,
        'C0': {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'F1': 0.0},
        'C1': {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'F1': 0.0},
        'C2': {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'F1': 0.0},
        'C3': {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'F1': 0.0},
    }

    # Combine
    for batch_eval in batch_eval_results:
        for key in batch_eval.keys():
            if not isinstance(eval_results[key], dict):
                eval_results[key] += batch_eval[key]  # acc_all
            else:
                # eval_results[key] = list(map(add, eval_results[key], batch_eval[key]))
                for dict_key in ['acc', 'prec', 'rec', 'F1']:
                    eval_results[key][dict_key] += batch_eval[key][dict_key]
    # Normalize
    for key in eval_results.keys():
        value = eval_results[key]
        if not isinstance(eval_results[key], dict):
            eval_results[key] = round(value / batch_num, 4) # acc_all
        else:
            # eval_results[key] = [round(v / batch_num, 4) for v in value]
            for dict_key in ['acc', 'prec', 'rec', 'F1']:
                eval_results[key][dict_key] = round(eval_results[key][dict_key] / batch_num, 4)

    return eval_results
