"""
Helper Functions Module

This module contains helper functions for data processing, I/O operations, and model utilities.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ============================================================================
# Data Processing Functions
# ============================================================================

def unzip_data(data, aug=True, aug_num=0):

    res = []
    
    if aug:
        for user in tqdm(data):
            user_seq = data[user]
            seq_len = len(user_seq)
            for i in range(aug_num+2, seq_len+1):
                res.append(user_seq[:i])
    else:
        for user in tqdm(data):
            user_seq = data[user]
            res.append(user_seq)

    return res


def unzip_data_with_user(data, aug=True, aug_num=0):

    res = []
    users = []
    user_id = 1
    
    if aug:
        for user in tqdm(data):
            user_seq = data[user]
            seq_len = len(user_seq)
            for i in range(aug_num+2, seq_len+1):
                res.append(user_seq[:i])
                users.append(user_id)
            user_id += 1
    else:
        for user in tqdm(data):
            user_seq = data[user]
            res.append(user_seq)
            users.append(user_id)
            user_id += 1

    return res, users


def concat_data(data_list):

    res = []

    if len(data_list) == 2:
        train = data_list[0]
        valid = data_list[1]
        for user in train:
            res.append(train[user]+valid[user])
    elif len(data_list) == 3:
        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]
        for user in train:
            res.append(train[user]+valid[user]+test[user])
    else:
        raise ValueError

    return res


def concat_aug_data(data_list):

    res = []
    train = data_list[0]
    valid = data_list[1]

    for user in train:
        if len(valid[user]) == 0:
            res.append([train[user][0]])
        else:
            res.append(train[user]+valid[user])

    return res


def concat_data_with_user(data_list):

    res = []
    users = []
    user_id = 1

    if len(data_list) == 2:
        train = data_list[0]
        valid = data_list[1]
        for user in train:
            res.append(train[user]+valid[user])
            users.append(user_id)
            user_id += 1
    elif len(data_list) == 3:
        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]
        for user in train:
            res.append(train[user]+valid[user]+test[user])
            users.append(user_id)
            user_id += 1
    else:
        raise ValueError

    return res, users


def filter_data(data, threshold=5):

    res = []
    for user in data:
        if len(user) > threshold:
            res.append(user)
        else:
            continue
    return res


def random_neq(l, r, s=[]):

    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def random_neq2(l, r, s=[], neg_num=1):

    candidates = set(range(l, r)) - set(s)
    neg_list = random.sample(list(candidates), neg_num)
    return np.array(neg_list)


# ============================================================================
# I/O Functions
# ============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def record_csv(args, res_dict, path='log'):

    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):
        df = pd.DataFrame(new_res_dict)
        df = df[columns]
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)


# ============================================================================
# Model Utility Functions
# ============================================================================

def get_n_params(model):

    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def load_pretrained_model(pretrain_dir, model, logger, device):

    logger.info("Loading pretrained model ... ")
    checkpoint_path = os.path.join(pretrain_dir, 'pytorch_model.bin')

    model_dict = model.state_dict()

    try:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    except:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)

    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)

    return model
