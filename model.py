import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
import numpy as np
import time
import utils
class BaseModel(nn.Module):
    def __init__(self, params, ent_tot, rel_tot):
        super(BaseModel, self).__init__()
        self.name = params.model
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = params.embedding_dim
        self.sigmoid_flag = params.sigmoid_flag
        self.mode = params.mode
        self.device = torch.device("cuda:" + str(params.cuda) if params.cuda > -1 else "cpu")
        if params.mode not in ['kvsall', 'neg_sample', '1vsall']:
            raise Exception('please choose correct training mode: kvsall, neg_sample')
        if params.loss == 'margin':
            self.loss = nn.MarginRankingLoss(margin=params.margin, reduction='mean')
        elif params.loss == 'bce':
            self.loss = nn.BCELoss(reduction='mean')
        elif params.loss == 'ce':
            self.loss = nn.CrossEntropyLoss(reduction='mean')
        else:
            raise Exception('loss function error: please choose loss function: bce, margin')
        self.loss = self.loss.to(self.device)

# class embedding(nn.Module):
#     def __init__(self, ent_tot, dim, padding_idx=0):
#         super(embedding, self).__init__()
#         self.emb = nn.Embedding(ent_tot, dim, padding_idx=padding_idx)
#     def init_weights(self, xv=True):
#         if xv:
#             nn.init.xavier_uniform_(self.emb.weight.data)
#         else:
#             nn.init.uniform_(self.emb.weight.data)



#     def forward(self, batch):
#         return self.emb(batch)
#     def save_emb(self, path):
#         torch.save(self.emb.state_dict(), path)
#     def load_emb(self, path):
#         self.emb.load_state_dict(torch.load(path))


class TransE(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(TransE, self).__init__(params, ent_tot, rel_tot)
        self.root_dir = 'checkpoint/TransE/' + params.data 
        self.params = params
        self.p_norm = params.p_norm
        self.ent_embeddings = nn.Embedding(ent_tot, self.dim, padding_idx=0)
        self.rel_embeddings = nn.Embedding(rel_tot, self.dim, padding_idx=0)
        self.ec_flag = False
        if params.entity_cluster_num > 0:
            # self.rel_indices = utils.get_rel_cluster_indices(params.data, params.embedding_dim, params.entity_cluster_num)
            self.indices = utils.get_ent_cluster_indices(params.data, params.embedding_dim, params.entity_cluster_num)
            self.entity_cluster_embed = nn.Embedding(params.entity_cluster_num, params.embedding_dim, padding_idx=0)
            # self.rel_cluster_embed = nn.Embedding(300, params.embedding_dim, padding_idx=0)
            self.ec_flag = True
        
        self.init_weights()
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        if self.params.entity_cluster_num > 0:
            nn.init.xavier_normal_(self.entity_cluster_embed.weight.data)
            # nn.init.xavier_normal_(self.rel_cluster_embed.weight.datas)
    
    def get_ent_clu_idx(self, ent):
        res = self.indices[ent.cpu().numpy()]
        return torch.Tensor(res).long().to('cuda')
    def get_rel_clu_idx(self, rel):
        res = self.rel_indices[rel.cpu().numpy()]
        return torch.Tensor(res).long().to('cuda')
    
    def save_embeddings(self):
        torch.save(self.ent_embeddings.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_ent_emb.pkl')
        torch.save(self.rel_embeddings.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_rel_emb.pkl')


    def _calc(self, h, r, t, predict=False):
        '''
            calculate ||h + r - t||_n 
        '''
        batch_h = self.ent_embeddings(h)
        batch_r = self.rel_embeddings(r)
        batch_t = self.ent_embeddings(t)
        if self.ec_flag:
            h_ = self.get_ent_clu_idx(h)
            t_ = self.get_ent_clu_idx(t)
            batch_h = (batch_h + self.entity_cluster_embed(h_)) / 2
            batch_t = (batch_t + self.entity_cluster_embed(t_)) / 2
            # r_ = self.get_rel_clu_idx(r)
            # batch_r = batch_r + self.rel_cluster_embed(r_)
        
        if self.mode == 'kvsall' or self.mode == '1vsall' or predict:
            emb_hr = batch_h + batch_r
            tsize = batch_h.shape[0]
            emb_hr = emb_hr.reshape((tsize, 1, -1)).repeat(1, self.ent_tot, 1)
            emb_c = self.ent_embeddings.weight.reshape((1, self.ent_tot, self.dim)).repeat(tsize, 1, 1)
            score = torch.norm(emb_hr - emb_c, self.p_norm, -1)
            if self.sigmoid_flag:
                score = torch.sigmoid(score)
        elif self.mode == 'neg_sample':
            score = torch.norm(batch_h + batch_r - batch_t, self.p_norm, -1)
            if self.sigmoid_flag:
                score = torch.sigmoid(score)
        return score

    def forward(self, h, r, t, batch_size):  
        
        score = self._calc(h, r, t) 
        neg_score = None
        if self.mode == 'neg_sample':
            pos_score = score[0: batch_size]
            neg_score = score[batch_size:]
        else:
            pos_score = score
        return pos_score, neg_score    

    def predict(self, h, r, t):
        score = self._calc(h, r, t, predict=True)
        return score


class DistMult(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(DistMult, self).__init__(params, ent_tot, rel_tot)
        self.p_norm = params.p_norm
        self.ent_embeddings = nn.Embedding(ent_tot, self.dim, padding_idx=0)
        self.rel_embeddings = nn.Embedding(rel_tot, self.dim, padding_idx=0)
        self.init_weights()
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
    def _calc(self, h, r, t, predict=False):
        batch_h = self.ent_embeddings(h)
        batch_r = self.rel_embeddings(r)
        batch_t = self.ent_embeddings(t)
        if self.mode == 'kvsall' or self.mode == '1vsall' or predict:
            emb_hr = batch_h*batch_r
            score = torch.mm(emb_hr, self.ent_embeddings.weight.transpose(1, 0))
        elif self.mode == 'neg_sample':
            score = batch_h*batch_r*batch_t
            score = torch.sum(score, -1)
        if self.sigmoid_flag:
                score = torch.sigmoid(score)
        return score

    def forward(self, h, r, t, batch_size):  
        score = self._calc(h, r, t) 
        neg_score = None
        if self.mode == 'neg_sample':
            pos_score = score[0: batch_size]
            neg_score = score[batch_size:]
        else:
            pos_score = score
        return pos_score, neg_score    

    def predict(self, h, r, t):
        score = self._calc(h, r, t, predict=True)
        return score
        


class ConvE(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(ConvE, self).__init__(params, ent_tot, rel_tot)
        self.ent_embeddings = torch.nn.Embedding(ent_tot, params.embedding_dim, padding_idx=0)
        self.rel_embeddings = torch.nn.Embedding(rel_tot, params.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(params.input_drop)
        self.hidden_drop = torch.nn.Dropout(params.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(params.feat_drop)
        self.emb_dim1 = params.embedding_shape1
        self.emb_dim2 = params.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=params.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(params.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(ent_tot)))
        self.fc = torch.nn.Linear(params.hidden_size,params.embedding_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.ent_embeddings.weight.data)
        nn.init.xavier_normal_(self.rel_embeddings.weight.data)
    
    def _calc(self, h, rel, t, predict=False):
        h_embed= self.ent_embeddings(h).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embed = self.rel_embeddings(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([h_embed, rel_embed], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        if self.mode == 'kvsall' or self.mode == '1vsall' or predict:
            x = torch.mm(x, self.ent_embeddings.weight.transpose(1,0))
            x += self.b.expand_as(x)
            score = x
            
        elif self.mode == 'neg_sample':
            t_embed = self.ent_embeddings(t)
            x = torch.sum(x*t_embed, -1)
            # print(x.shape)
            tmp_b = self.b[h]
            # print(tmp_b.shape)
            x += tmp_b
            score = x
        
        if self.sigmoid_flag:
            score = torch.sigmoid(x)
        return score

    def forward(self, h, r, t, batch_size):  
        score = self._calc(h, r, t) 
        neg_score = None
        # print(score.shape)
        if self.mode == 'neg_sample':
            pos_score = score[0: batch_size]
            neg_score = score[batch_size:]
        else:
            pos_score = score
        return pos_score, neg_score    

    def predict(self, h, r, t):
        score = self._calc(h, r, t, predict=True)
        return score
        

class ComplEx(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(ComplEx, self).__init__(params, ent_tot, rel_tot)
        self.emb_e_real = torch.nn.Embedding(ent_tot, params.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(ent_tot, params.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(rel_tot, params.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(rel_tot, params.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(params.input_drop)
        self.init_weights()
    def init_weights(self):
        nn.init.xavier_normal_(self.emb_e_real.weight.data)
        nn.init.xavier_normal_(self.emb_e_img.weight.data)
        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_img.weight.data)
    
    def _calc(self, h, r, t, predict=False):
        h_embedding_real = self.emb_e_real(h)
        r_embedding_real = self.emb_rel_real(r)
        h_embedding_img = self.emb_e_img(h)
        r_embedding_img = self.emb_rel_img(r)

        h_embedding_real = self.inp_drop(h_embedding_real)
        r_embedding_real = self.inp_drop(r_embedding_real)
        h_embedding_img = self.inp_drop(h_embedding_img)
        r_embedding_img = self.inp_drop(r_embedding_img)
        # print(h_embedding_real.shape, r_embedding_real.shape)
        if self.mode == 'kvsall' or self.mode == '1vsall' or predict:
            realrealreal = torch.mm(h_embedding_real*r_embedding_real, self.emb_e_real.weight.transpose(1,0))
            realimgimg = torch.mm(h_embedding_real*r_embedding_img, self.emb_e_img.weight.transpose(1,0))
            imgrealimg = torch.mm(h_embedding_img*r_embedding_real, self.emb_e_img.weight.transpose(1,0))
            imgimgreal = torch.mm(h_embedding_img*r_embedding_img, self.emb_e_real.weight.transpose(1,0))
            score = realrealreal  + realimgimg + imgrealimg - imgimgreal
            
        elif self.mode == 'neg_sample':
            t_embedding_real = self.emb_e_real(t)
            t_embedding_img = self.emb_e_img(t)
            realrealreal = h_embedding_real*r_embedding_real*t_embedding_real
            realimgimg = h_embedding_real*r_embedding_img*t_embedding_img
            imgrealimg = h_embedding_img*r_embedding_real*t_embedding_img
            imgimgreal = h_embedding_img*r_embedding_img*t_embedding_real
            score = realrealreal  + realimgimg + imgrealimg - imgimgreal
            score = torch.sum(score, -1)
        if self.sigmoid_flag:
            score = torch.sigmoid(score)
        return score
    def forward(self, h, r, t, batch_size):  
        score = self._calc(h, r, t) 
        neg_score = None
        if self.mode == 'neg_sample':
            pos_score = score[0: batch_size]
            neg_score = score[batch_size:]
        else:
            pos_score = score
        return pos_score, neg_score    

    def predict(self, h, r, t):
        score = self._calc(h, r, t, predict=True)
        return score