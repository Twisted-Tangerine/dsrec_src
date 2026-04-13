"""
Evaluation Metrics Module

This module contains functions for computing recommendation evaluation metrics,
including NDCG, HR, and group-based metrics for different user/item segments.
"""

import numpy as np


def metric_report(data_rank, topk=10):

    NDCG, HT = 0, 0
    
    for rank in data_rank:
        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {'NDCG@10': NDCG / len(data_rank),
            'HR@10': HT / len(data_rank)}


def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):

    if args is not None:
        ts_user = args.threshold_user
    else:
        ts_user = 10

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    count_s = len(data_len[data_len<ts_user+aug_len])
    count_l = len(data_len[data_len>=ts_user+aug_len])

    for i, rank in enumerate(data_rank):
        if rank < topk:
            if data_len[i] < ts_user+aug_len:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0,
            'Short HR@10': HT_s / count_s if count_s!=0 else 0,
            'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Long HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_pop_report(data_rank, pop_dict, target_items, topk=10, aug_pop=0, args=None):

    if args is not None:
        ts_tail = args.threshold_item
    else:
        ts_tail = 20

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    item_pop = pop_dict[target_items.astype("int64")]
    count_s = len(item_pop[item_pop<ts_tail+aug_pop])
    count_l = len(item_pop[item_pop>=ts_tail+aug_pop])

    for i, rank in enumerate(data_rank):
        if i == 0:  # skip the padding index
            continue

        if rank < topk:
            if item_pop[i] < ts_tail+aug_pop:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Tail NDCG@10': NDCG_s / count_s if count_s!=0 else 0,
            'Tail HR@10': HT_s / count_s if count_s!=0 else 0,
            'Popular NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Popular HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_len_5group(pred_rank, seq_len, thresholds=[5, 10, 15, 20], topk=10):

    NDCG = np.zeros(5)
    HR = np.zeros(5)
    
    for i, rank in enumerate(pred_rank):
        target_len = seq_len[i]
        if rank < topk:
            if target_len < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1
            elif target_len < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1
            elif target_len < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1
            elif target_len < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1
            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
    count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
    count[2] = len(seq_len[seq_len>=thresholds[1]]) - len(seq_len[seq_len>=thresholds[2]])
    count[3] = len(seq_len[seq_len>=thresholds[2]]) - len(seq_len[seq_len>=thresholds[3]])
    count[4] = len(seq_len[seq_len>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j] if count[j] > 0 else 0
        HR[j] = HR[j] / count[j] if count[j] > 0 else 0

    return HR, NDCG, count


def metric_pop_5group(pred_rank, pop_dict, target_items, thresholds=[10, 30, 60, 100], topk=10):

    NDCG = np.zeros(5)
    HR = np.zeros(5)
    
    for i, rank in enumerate(pred_rank):
        target_pop = pop_dict[int(target_items[i])]
        if rank < topk:
            if target_pop < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1
            elif target_pop < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1
            elif target_pop < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1
            elif target_pop < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1
            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    pop = pop_dict[target_items.astype("int64")]
    count[0] = len(pop[pop>=0]) - len(pop[pop>=thresholds[0]])
    count[1] = len(pop[pop>=thresholds[0]]) - len(pop[pop>=thresholds[1]])
    count[2] = len(pop[pop>=thresholds[1]]) - len(pop[pop>=thresholds[2]])
    count[3] = len(pop[pop>=thresholds[2]]) - len(pop[pop>=thresholds[3]])
    count[4] = len(pop[pop>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j] if count[j] > 0 else 0
        HR[j] = HR[j] / count[j] if count[j] > 0 else 0

    return HR, NDCG, count


def seq_acc(true, pred):

    true_num = np.sum((true==pred))
    total_num = true.shape[0] * true.shape[1]
    return {'acc': true_num / total_num}
