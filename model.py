import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
import numpy as np
import time
import utils

class SoftplusLoss(nn.Module):
	def __init__(self):
		super(SoftplusLoss, self).__init__()
		self.criterion = nn.Softplus()

	def forward(self, p_score, n_score):
		return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2
			
        
class BaseModel(nn.Module):
    def __init__(self, params, ent_tot, rel_tot):
        super(BaseModel, self).__init__()
        self.params = params
        self.model_name = params.model
        self.data_name = params.data
        self.root_dir = 'checkpoint/{}/{}'.format(self.model_name, self.data_name)
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
        elif params.loss == 'sfmargin':
            self.loss = nn.SoftMarginLoss(reduction='mean')
        elif params.loss == 'sploss':
            self.loss = SoftplusLoss()
        else:
            raise Exception('loss function error: please choose loss function: bce, margin, ce, sfmargin')
        self.loss = self.loss.to(self.device)
        self.ec_flag = False
        self.rc_flag = False
        
        if params.cluster_ent_name != '':
            self.ent_indices = np.array(utils.get_ent_cluster_indices('cluster/' + params.cluster_ent_name + '.pkl'))
            cluster_num = max(self.ent_indices) + 1
            
            self.entity_cluster_embed = nn.Embedding(cluster_num, params.embedding_dim)
            self.ec_flag = True
            nn.init.constant_(self.entity_cluster_embed.weight.data, 0.0)
            # nn.init.xavier_uniform_(self.entity_cluster_embed.weight.data)
            # self.entity_cluster_embed_mul = nn.Embedding(cluster_num, params.embedding_dim)
            # nn.init.constant_(self.entity_cluster_embed_mul.weight.data, 1.0)
            # nn.init.xavier_uniform_(self.entity_cluster_embed_mul.weight.data)
        if params.cluster_rel_name != '':
            self.rel_indices = np.array(utils.get_rel_cluster_indices('cluster/' + params.cluster_rel_name + '.pkl'))
            cluster_num = max(self.rel_indices) + 1
            self.rel_cluster_embed = nn.Embedding(cluster_num, params.embedding_dim)
            
            # self.relc2idx = [[] for i in range(cluster_num)]
            # for n,i in enumerate(self.rel_indices):
            #     self.relc2idx[i].append(n)
            # print(self.relc2idx)
            self.rc_flag = True
            nn.init.constant_(self.rel_cluster_embed.weight.data, 0.0)
            # nn.init.xavier_uniform_(self.rel_cluster_embed.weight.data)
            # self.rel_cluster_embed_mul = nn.Embedding(cluster_num, params.embedding_dim)
            # nn.init.constant_(self.rel_cluster_embed_mul.weight.data, 1.0)
            # nn.init.xavier_uniform_(self.rel_cluster_embed_mul.weight.data)
    def save_embeddings(self):
        torch.save(self.ent_embeddings.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_ent_emb.pkl')
        torch.save(self.rel_embeddings.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_rel_emb.pkl')

    def get_ent_clu_idx(self, ent):
        res = self.ent_indices[ent.cpu().numpy()]
        return torch.Tensor(res).long().to(self.device)
    def get_rel_clu_idx(self, rel):
        res = self.rel_indices[rel.cpu().numpy()]
        return torch.Tensor(res).long().to(self.device)
    def forward(self, h, r, t, batch_size):  
        
        score = self._calc(h, r, t) 
        neg_score = None
        if self.mode == 'neg_sample':
            pos_score = score[0: batch_size]
            neg_score = score[batch_size:]
        else:
            pos_score = score
        return pos_score, neg_score 

    def predict(self, h, r, t, isEval):
        score = self._calc(h, r, t, isEval)
        return score
    def regul(self, h, r, t):
        batch_h = self.ent_embeddings(h)
        batch_r = self.rel_embeddings(r)
        batch_t = self.ent_embeddings(t)
        return (torch.mean(batch_h**2) + torch.mean(batch_r**2) + torch.mean(batch_t**2))/3
        # return torch.mean(batch_r**2)
    def regul_r(self, h, r, t):
        
        if self.params.cluster_rel_name != '':
            r_ = self.get_rel_clu_idx(r)
            batch_r_c = self.rel_cluster_embed(r_)
        return torch.mean(batch_r_c**2)
    def regul_e(self, h, r, t):
        if self.params.cluster_ent_name != '':
            h_ = self.get_ent_clu_idx(h)
            t_ = self.get_ent_clu_idx(t)
            batch_h_c = self.entity_cluster_embed(h_)
            batch_t_c = self.entity_cluster_embed(t_)
        return torch.mean(batch_h_c**2) + torch.mean(batch_t_c**2)



class TransE(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(TransE, self).__init__(params, ent_tot, rel_tot)
        self.params = params
        self.p_norm = params.p_norm
        self.ent_embeddings = nn.Embedding(ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(rel_tot, self.dim)

        #debug

        
        self.init_weights()
    def init_weights(self):
        
        # if self.params.cluster_ent_name or self.params.cluster_rel_name:
        #     print('loading preTrain embedding...')
        #     self.ent_embeddings.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_ent_emb.pkl'.format(self.model_name, self.data_name, self.dim)))
        #     self.rel_embeddings.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_rel_emb.pkl'.format(self.model_name, self.data_name, self.dim)))
        # else:  
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
    # def regul(self, h, r, t):
    #     batch_h = self.ent_embeddings(h)
    #     batch_r = self.rel_embeddings(r)
    #     batch_t = self.ent_embeddings(t)
    #     return (torch.mean(batch_h, 2) + torch.mean(batch_r**2) + torch.mean(batch_t**2))/3


    def _calc(self, h, r, t, isEval=False):
        '''
            calculate ||h + r - t||_n 
        '''
        batch_h = self.ent_embeddings(h)
        batch_r = self.rel_embeddings(r)
        batch_t = self.ent_embeddings(t)
        if self.ec_flag:
            h_ = self.get_ent_clu_idx(h)
            t_ = self.get_ent_clu_idx(t)
            batch_h = batch_h + self.params.b2*self.entity_cluster_embed(h_)
            batch_t = batch_t + self.params.b2*self.entity_cluster_embed(t_)
            


        if self.rc_flag:
            r_ = self.get_rel_clu_idx(r)
            batch_r = batch_r + self.params.b1*self.rel_cluster_embed(r_)
        if self.params.norm_flag1:
            batch_h = F.normalize(batch_h, 2, -1)
            batch_r = F.normalize(batch_r, 2, -1)
            batch_t = F.normalize(batch_t, 2, -1)
        if (self.mode == 'kvsall' or self.mode == '1vsall') and (not isEval):
            emb_hr = batch_h + batch_r
            tsize = batch_h.shape[0]
            emb_hr = emb_hr.reshape((tsize, 1, -1)).repeat(1, self.ent_tot, 1)
            emb_c = self.ent_embeddings.weight.reshape((1, self.ent_tot, self.dim)).repeat(tsize, 1, 1)
            score = torch.norm(emb_hr - emb_c, self.p_norm, -1)
            if self.sigmoid_flag:
                score = torch.sigmoid(score)
        elif self.mode == 'neg_sample' or isEval:
            score = torch.norm(batch_h + batch_r - batch_t, self.p_norm, -1)
        return score

    


class DistMult(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(DistMult, self).__init__(params, ent_tot, rel_tot)
        # self.p_norm = params.p_norms
        # self.norm_flag = prarams.norm_flag
        self.ent_embeddings = nn.Embedding(ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(rel_tot, self.dim)
        self.norm = nn.BatchNorm1d(self.dim)
        self.init_weights()
    def save_embeddings(self):
        torch.save(self.ent_embeddings.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_ent_emb.pkl')
        torch.save(self.rel_embeddings.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_rel_emb.pkl')
        if self.params.norm_flag1:
            torch.save(self.norm.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_norm.pkl')
    def init_weights(self):
        # if self.params.cluster_ent_name or self.params.cluster_rel_name:
        # print('loading preTrain embedding...')
        # self.ent_embeddings.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_ent_emb.pkl'.format(self.model_name, self.data_name, self.dim)))
        # self.rel_embeddings.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_rel_emb.pkl'.format(self.model_name, self.data_name, self.dim)))
            # self.ent_embeddings.load_state_dict(torch.load('checkpoint/TransE/{}/dim_{}_ent_emb.pkl'.format(self.data_name, self.dim)))
            # self.rel_embeddings.load_state_dict(torch.load('checkpoint/TransE/{}/dim_{}_rel_emb.pkl'.format(self.data_name, self.dim)))
        # else:
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        # if self.params.norm_flag1:
        #     self.norm.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_norm.pkl'.format(self.model_name, self.data_name, self.dim)))

    def _calc(self, h, r, t, isEval=False):
        batch_h = self.ent_embeddings(h)
        batch_r = self.rel_embeddings(r)
        batch_t = self.ent_embeddings(t)
        if self.ec_flag:
            h_ = self.get_ent_clu_idx(h)
            t_ = self.get_ent_clu_idx(t)
            batch_h = batch_h + self.params.b2*self.entity_cluster_embed(h_)
            batch_t = batch_t + self.params.b2*self.entity_cluster_embed(t_)
        if self.rc_flag:
            r_ = self.get_rel_clu_idx(r)
            batch_r = batch_r + self.params.b1*self.rel_cluster_embed(r_)

        if (self.mode == 'kvsall' or self.mode == '1vsall') and (not isEval):
            emb_hr = batch_h*batch_r
            score = torch.mm(emb_hr, self.ent_embeddings.weight.transpose(1, 0))
        elif self.mode == 'neg_sample' or isEval:
            score = batch_h*batch_r*batch_t
            if self.params.norm_flag1:
                score = self.norm(score)
            score = torch.sum(score, -1)

        if self.sigmoid_flag:
            score = torch.sigmoid(score)

        if self.params.norm_flag2:
            score = F.normalize(score, 2, -1)
        # score[score > 10.0] = 10.0
        
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
    


        


class ConvE(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(ConvE, self).__init__(params, ent_tot, rel_tot)
        self.ent_embeddings = torch.nn.Embedding(ent_tot, params.embedding_dim)
        self.rel_embeddings = torch.nn.Embedding(rel_tot, params.embedding_dim)
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
    
    def _calc(self, h, rel, t, isEval=False):
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
        if (self.mode == 'kvsall' or self.mode == '1vsall' ) and (not isEval):
            x = torch.mm(x, self.ent_embeddings.weight.transpose(1,0))
            x += self.b.expand_as(x)
            score = x
            
        elif self.mode == 'neg_sample'  or isEval:
            t_embed = self.ent_embeddings(t)
            x = torch.sum(x*t_embed, -1)
            # print(x.shape)
            tmp_b = self.b[h]
            # print(tmp_b.shape)
            x += tmp_b
            # score = F.normalize(x, 2, -1)
        
        if self.sigmoid_flag:
            score = torch.sigmoid(x)
        return -score

   
        

class ComplEx(BaseModel):
    def __init__(self, params, ent_tot, rel_tot):
        super(ComplEx, self).__init__(params, ent_tot, rel_tot)
        self.emb_e_real = torch.nn.Embedding(ent_tot, params.embedding_dim)
        self.emb_e_img = torch.nn.Embedding(ent_tot, params.embedding_dim)
        self.emb_rel_real = torch.nn.Embedding(rel_tot, params.embedding_dim)
        self.emb_rel_img = torch.nn.Embedding(rel_tot, params.embedding_dim)
        self.inp_drop = torch.nn.Dropout(params.input_drop)
        self.norm = nn.BatchNorm1d(self.dim)
        self.init_weights()
        if params.cluster_ent_name != '':
            cluster_num = max(self.ent_indices) + 1
            print(cluster_num)
            self.entity_cluster_embed_img = nn.Embedding(cluster_num, params.embedding_dim)
            nn.init.constant_(self.entity_cluster_embed_img.weight.data, 0)
        if params.cluster_rel_name != '':
            cluster_num = max(self.rel_indices) + 1
            print(cluster_num)
            self.rel_cluster_embed_img = nn.Embedding(cluster_num, params.embedding_dim)
            nn.init.constant_(self.rel_cluster_embed_img.weight.data, 0)
        
    def init_weights(self):
        # if self.params.cluster_ent_name or self.params.cluster_rel_name:
        #     print('loading preTrain embedding...')
        #     self.emb_e_real.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_ent_emb.pkl'.format(self.model_name, self.data_name, self.dim)))
        #     self.emb_rel_real.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_rel_emb.pkl'.format(self.model_name, self.data_name, self.dim)))
        #     self.emb_e_img.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_ent_emb_img.pkl'.format(self.model_name, self.data_name, self.dim)))
        #     self.emb_rel_img.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_rel_emb_img.pkl'.format(self.model_name, self.data_name, self.dim)))
        # else:
        nn.init.xavier_uniform_(self.emb_e_real.weight.data)
        nn.init.xavier_uniform_(self.emb_e_img.weight.data)
        nn.init.xavier_uniform_(self.emb_rel_real.weight.data)
        nn.init.xavier_uniform_(self.emb_rel_img.weight.data)
        # if self.params.norm_flag1:
        #     self.norm.load_state_dict(torch.load('checkpoint/{}/{}/dim_{}_norm.pkl'.format(self.model_name, self.data_name, self.dim)))

    def save_embeddings(self):
        torch.save(self.emb_e_real.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_ent_emb.pkl')
        torch.save(self.emb_rel_real.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_rel_emb.pkl')
        torch.save(self.emb_e_img.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_ent_emb_img.pkl')
        torch.save(self.emb_rel_img.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_rel_emb_img.pkl')
        if self.params.norm_flag1:
            torch.save(self.norm.state_dict(), self.root_dir + '/dim_' + str(self.params.embedding_dim) + '_norm.pkl')
    def regul(self, h, r, t):
        batch_h_real = self.emb_e_real(h) 
        batch_r_real = self.emb_rel_real(r) 
        batch_t_real = self.emb_e_real(t) 
        batch_h_img = self.emb_e_img(h)
        batch_r_img = self.emb_rel_img(r)
        batch_t_img = self.emb_e_img(t)
        # if self.rc_flag:
        #     r_ = self.get_rel_clu_idx(r)
        #     batch_r +=  self.rel_cluster_embed(r_) + self.rel_cluster_embed_img(r_)
        # if self.ec_flag:
        #     h_ = self.get_ent_clu_idx(h)
        #     t_ = self.get_ent_clu_idx(t)
        #     batch_h += self.entity_cluster_embed(h_) + self.entity_cluster_embed_img(h_)
        #     batch_t += self.entity_cluster_embed(t_) + self.entity_cluster_embed_img(t_)

        return (torch.mean(batch_h_real**2) + torch.mean(batch_r_real**2) + torch.mean(batch_t_real**2) + \
                torch.mean(batch_h_img**2) + torch.mean(batch_r_img**2) + torch.mean(batch_t_img**2))/6
    def _calc(self, h, r, t, isEval=False):
        h_embedding_real = self.emb_e_real(h)
        r_embedding_real = self.emb_rel_real(r)
        h_embedding_img = self.emb_e_img(h)
        r_embedding_img = self.emb_rel_img(r)


        # h_embedding_real = self.inp_drop(h_embedding_real)
        # r_embedding_real = self.inp_drop(r_embedding_real)
        # h_embedding_img = self.inp_drop(h_embedding_img)
        # r_embedding_img = self.inp_drop(r_embedding_img)
        # print(h_embedding_real.shape, r_embedding_real.shape)
        if (self.mode == 'kvsall' or self.mode == '1vsall') and not isEval:
            realrealreal = torch.mm(h_embedding_real*r_embedding_real, self.emb_e_real.weight.transpose(1,0))
            realimgimg = torch.mm(h_embedding_real*r_embedding_img, self.emb_e_img.weight.transpose(1,0))
            imgrealimg = torch.mm(h_embedding_img*r_embedding_real, self.emb_e_img.weight.transpose(1,0))
            imgimgreal = torch.mm(h_embedding_img*r_embedding_img, self.emb_e_real.weight.transpose(1,0))
            score = realrealreal  + realimgimg + imgrealimg - imgimgreal
            
        elif self.mode == 'neg_sample'  or isEval:
            t_embedding_real = self.emb_e_real(t)
            t_embedding_img = self.emb_e_img(t)
            if self.rc_flag:
                r_ = self.get_rel_clu_idx(r)
                r_embedding_real = r_embedding_real + self.params.b1*self.rel_cluster_embed(r_)
                r_embedding_img = r_embedding_img + self.params.b1*self.rel_cluster_embed_img(r_)
            if self.ec_flag:
                h_ = self.get_ent_clu_idx(h)
                t_ = self.get_ent_clu_idx(t)
                h_embedding_real = h_embedding_real + self.params.b2*self.entity_cluster_embed(h_)
                h_embedding_img = h_embedding_img + self.params.b2*self.entity_cluster_embed_img(h_)
                t_embedding_real = t_embedding_real + self.params.b2*self.entity_cluster_embed(t_)
                t_embedding_img = t_embedding_img + self.params.b2*self.entity_cluster_embed_img(t_)

            
            realrealreal = h_embedding_real*r_embedding_real*t_embedding_real
            realimgimg = h_embedding_real*r_embedding_img*t_embedding_img
            imgrealimg = h_embedding_img*r_embedding_real*t_embedding_img
            imgimgreal = h_embedding_img*r_embedding_img*t_embedding_real
            score = realrealreal  + realimgimg + imgrealimg - imgimgreal
            if self.params.norm_flag1:
                score = self.norm(score)
            score = torch.sum(score, -1)
        if self.sigmoid_flag:
            score = torch.sigmoid(score)
        # score = F.normalize(score, 2, -1)
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