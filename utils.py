import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from config import *
from data.dataLoader import *
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import pickle
import numpy as np



def get_save_model_path(params):
    data = params.data
    model = params.model
    optim = params.optimizer
    lr = str(params.lr)
    mode = params.mode
    margin = str(params.margin)
    p_norm = str(params.p_norm)
    emb_dim = str(params.embedding_dim)

    loss = params.loss
    name = '_'.join([optim, lr, mode, loss, 'embdim', emb_dim])
    print(name)
    



def dataset_param(dataset_name):
    return datasets_param.d[dataset_name]['ent_tot'], datasets_param.d[dataset_name]['rel_tot']

def get_optimizer(model, params):
    if params.optimizer == 'SGD':
        opt = optim.SGD(model.parameters(), params.lr, params.momentum, weight_decay=params.regularize)
    elif params.optimizer == 'Adam':
        opt = optim.Adam(model.parameters(), params.lr, weight_decay=params.regularize)
    elif params.optimizer == 'AdaGrad':
        opt = optim.Adagrad(model.parameters(), params.lr, weight_decay=params.regularize)
    else:
        raise Exception('please choose correct optimizer: SGD, Adam, AdaGrad')
    return opt

def get_model(params):
    print('[getting model {}]'.format(params.model))
    ent_tot, rel_tot = dataset_param(params.data)
    model_name = params.model
    print('# ent_tot:{}, rel_tot:{}, em_dim:{}'.format(ent_tot, rel_tot, params.embedding_dim))
    if model_name == 'TransE':
        return TransE(params, ent_tot=ent_tot, rel_tot=rel_tot)
    elif model_name == 'DistMult':
        return DistMult(params, ent_tot=ent_tot, rel_tot=rel_tot)
    elif model_name == 'ComplEx':
        return ComplEx(params, ent_tot=ent_tot, rel_tot=rel_tot)
    elif model_name == 'ConvE':
        return ConvE(params, ent_tot=ent_tot, rel_tot=rel_tot)
    else:
        raise Exception("please choose model from TransE/DistMult/ComplEx/ConvE.")

def get_loss(loss_name, margin):
    print('[getting loss function {}]'.format(loss_name))
    if loss_name == 'margin':
        return nn.MarginRankingLoss(margin=margin, reduction='sum')
    elif loss_name == 'bce':
        return nn.BCELoss()


def get_data_loader(params, filename_prefix='train'):
    dataset_name = params.data
    if filename_prefix not in ['train', 'valid']:
        mtype = 'test'
    else:
        mtype = filename_prefix
    ent_tot = datasets_param.d[dataset_name]['ent_tot']
    rel_tot = datasets_param.d[dataset_name]['rel_tot']
    root = datasets_param.d[dataset_name]['root'] 
    print('[loading {} data {}]'.format(filename_prefix, dataset_name), end=' ')
    
    tmp_loader = kge_data_loader(params, root, filename_prefix + '.pkl', ent_tot, mode=mtype)

    print('[ok]')
    if filename_prefix == 'train':
        return DataLoader(tmp_loader, batch_size=params.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        return DataLoader(tmp_loader, batch_size=params.test_batch_size, shuffle=True, num_workers=4, pin_memory=True)


def get_ent_cluster_indices(dataset, dim, c):
    fr = open('entity_cluster.pkl', 'rb')
    pk = pickle.load(fr)
    fr.close()
    return pk['fb']['dim_' + '100' + '_c_' + str(c)]

def get_rel_cluster_indices(dataset, dim, c):
    fr = open('cluster_relation.pkl', 'rb')
    pk = pickle.load(fr)
    fr.close()
    return pk['fb']['rel_c_300']

'''
 emb is nn.Embedding.

'''
def cal_ent_cluster_labels(emb, n_c):
    np_emb = emb.weight.data.numpy()
    estimator = KMeans(n_clusters=n_c)
    estimator.fit(np_emb)
    return estimator.labels_


def create_entity_cluster(embed_set, cluster_C_set = [[500, 1000, 2000, 4000], [1000, 2000, 4000, 8000]]):
    fb_ent_emb = embed_set[0]
    wn_ent_emb = embed_set[1]
    C_1 = cluster_C_set[0]
    C_2 = cluster_C_set[1]
    tmp = {'fb':{}, 'wn':{}}
    for i in C_1:
        tmp['fb']['dim_100_c_' + str(i)] = get_ent_cluster_labels(fb_ent_emb, i)
    for i in C_2:
        tmp['wn']['dim_100_c_' + str(i)] = get_ent_cluster_labels(wn_ent_emb, i)
    fw = open('entity_cluster.pkl', 'wb')
    pickle.dump(tmp, fw)

def create_entity_sub_cluster():
    pass