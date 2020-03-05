import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from config import *
from data.dataLoader import *
from torch.utils.data import DataLoader



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




