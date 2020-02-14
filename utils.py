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


def get_model(model_name, dataset_name, em_dim, p_norm):
    print('[getting model {}]'.format(model_name))
    ent_tot, rel_tot = dataset_param(dataset_name)
    print('# ent_tot:{}, rel_tot:{}, em_dim:{}'.format(ent_tot, rel_tot, em_dim))
    if model_name == 'TransE':
        return TransE(ent_tot=ent_tot, rel_tot=rel_tot, em_dim=em_dim, p_norm=p_norm)
    elif model_name == 'DistMult':
        return DistMult(ent_tot=ent_tot, rel_tot=rel_tot, em_dim=em_dim)
    elif model_name == 'adv_TransE':
        return adv_TransE(ent_tot=ent_tot, rel_tot=rel_tot, em_dim=em_dim)
    elif model_name == 'adv_DistMult':
        return adv_DistMult(ent_tot=ent_tot, rel_tot=rel_tot, em_dim=em_dim)
    else:
        pass

def get_loss(loss_name, margin):
    print('[getting loss function {}]'.format(loss_name))
    if loss_name == 'margin':
        return nn.MarginRankingLoss(margin=margin, reduction='sum')
    elif loss_name == 'bce':
        return nn.BCELoss()


def get_data_loader(dataset_name, batch_size, type='train', sample_flag=True, sample_size=1):
    ent_tot = datasets_param.d[dataset_name]['ent_tot']
    rel_tot = datasets_param.d[dataset_name]['rel_tot']
    root = datasets_param.d[dataset_name]['root'] 
    print('[loading data {}]'.format(dataset_name))
    
    if type in ['train', 'test', 'valid']:
        tmp_loader = kge_data_loader(root, type + '.txt', ent_tot, sample_flag, sample_size)
    else:
        tmp_loader = kge_data_loader(root, type + '.txt', ent_tot, sample_flag, sample_size)
    print('[ok]')

    return DataLoader(tmp_loader, batch_size=batch_size, shuffle=True)




