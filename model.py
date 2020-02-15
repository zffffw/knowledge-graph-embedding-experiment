import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import codecs
class TransE(nn.Module):
    def __init__(self, ent_tot, rel_tot, em_dim = 100, p_norm = 2, norm_flag = False, sigmoid_flag = False):
        super(TransE, self).__init__()
        self.name = 'TransE'
        self.dim = em_dim
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.p_norm = p_norm
        self.norm_flag = norm_flag
        self.ent_embeddings = nn.Embedding(ent_tot, em_dim, padding_idx=0)
        self.rel_embeddings = nn.Embedding(rel_tot, em_dim, padding_idx=0)
        self.sigmoid_flag = sigmoid_flag

        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        


    def _calc(self, h, r):
        '''
            calculate ||h + r - t||_n 
        '''
        emb_hr = h + r
        tsize = h.shape[0]
        emb_hr = emb_hr.reshape((tsize, 1, -1)).repeat(1, self.ent_tot, 1)
        
        emb_c = self.ent_embeddings.weight.reshape((1, self.ent_tot, self.dim)).repeat(tsize, 1, 1)
        # print('emb_hr:{}'.format(emb_hr.shape))
        # print('emb_c:{}'.format(emb_c.shape))
        score = torch.norm(emb_hr - emb_c, self.p_norm, -1)
        # print(score.shape)
        if self.sigmoid_flag:
            score = torch.sigmoid(score)
        # print(score.shape)
        return score

    def forward(self, h, r, t, batch_size):  
        batch_h = self.ent_embeddings(h)
        batch_r = self.rel_embeddings(r)
        batch_t = self.ent_embeddings(t)
        score = self._calc(batch_h, batch_r) 
        # print(score.shape)  
        pos_score = score[0: batch_size]
        neg_score = score[batch_size:]
        # print(pos_score.shape, neg_score.shape)
        return pos_score, neg_score    

    def predict(self, h, r):
        batch_h = self.ent_embeddings(h)
        batch_r = self.rel_embeddings(r)
        score = self._calc(batch_h, batch_r)       
        return score




class DistMult(nn.Module):
    def __init__(self, ent_tot, rel_tot, em_dim = 50, sigmoid_flag=True):
        super(DistMult, self).__init__()
        self.ent_embeddings = nn.Embedding(ent_tot, em_dim, padding_idx=0)
        self.rel_embeddings = nn.Embedding(rel_tot, em_dim, padding_idx=0)

        # self.criterion = nn.Softplus()
        self.sigmoid_flag = sigmoid_flag
        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, r):
        emb_hr = h*r
        # print('111', emb_hr.shape, self.ent_embeddings.weight.transpose(1, 0).shape)
        pred = torch.mm(emb_hr, self.ent_embeddings.weight.transpose(1, 0))
        if self.sigmoid_flag:
            score = torch.sigmoid(pred)
        return score

    def forward(self, batch_h, batch_r, batch_t, batch_size):
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        # t = self.ent_embeddings(batch_t)
        #y = torch.from_numpy(batch_y).type(torch.FloatTensor)

        score = self._calc(h, r)

        pos_score = score[0: batch_size]
        neg_score = score[batch_size: len(score)]

        # regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        # loss = torch.mean(self.criterion(score * y)) + self.params.lmbda * regul
        # loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]))
        
        return pos_score, neg_score
    def predict(self, batch_h, batch_r):
        h = self.ent_embeddings(batch_h)
        # t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        #y = torch.from_numpy(batch_y).type(torch.FloatTensor)

        score = self._calc(h, r)

        # regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        # loss = torch.mean(self.criterion(score * y)) + self.params.lmbda * regul
        # loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]))
        
        return score


    
class ConvE(nn.Module):
    def __init__(self, ent_tot, rel_tot, em_dim = 100, dropout=False, dropout_p = 0.0):
        super(ConvE, self).__init__()
        self.ent_embeddings = nn.Embedding(ent_tot, em_dim, padding_idx=0)
        self.rel_embeddings = nn.Embedding(rel_tot, em_dim, padding_idx=0)

        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

class ComplEx(nn.Module):
    def __init__(self, ent_tot, rel_tot, em_dim = 100, input_drop=0.2):
        super(ComplEx, self).__init__()
        self.emb_e_real = torch.nn.Embedding(ent_tot, em_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(ent_tot, em_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(rel_tot, em_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(rel_tot, em_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(input_drop)

'''
   # delete

class adv_TransE(nn.Module):
    def __init__(self, ent_tot, rel_tot, em_dim = 100):
        super(adv_TransE, self).__init__()
        self.name = 'adv_TransE'
        self.dim = em_dim
        self.ent_embeddings = nn.Embedding(ent_tot, em_dim, max_norm = 1)
        self.Vr = nn.Embedding(rel_tot, em_dim)
        self.W = nn.Parameter(torch.zeros((em_dim, em_dim)), requires_grad=True)
        self.init_weights()
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.Vr.weight.data)
        nn.init.xavier_uniform_(self.W)
    def _calc(self, batch_h, batch_r, batch_t):
        # yh = torch.sigmoid(torch.mm(self.W, batch_h.T).T)
        # yt = torch.sigmoid(torch.mm(self.W, batch_t.T).T)
        yh = batch_h
        yt = batch_t
        # yh = torch.mm(self.W, batch_h.T).T
        # yt = torch.mm(self.W, batch_t.T).T
        # print(yh.shape, yt.shape, batch_r.shape)
        return torch.norm(yh - yt + batch_r, 2, -1)
    def forward(self, batch_h, batch_r, batch_t, batch_size):
        batch_h = self.ent_embeddings(batch_h)
        batch_r = self.Vr(batch_r)
        batch_t = self.ent_embeddings(batch_t)
        score = self._calc(batch_h, batch_r, batch_t)
        pos_score = score[:batch_size]
        neg_score = score[batch_size:]
        return pos_score, neg_score
    def predict(self, h, r, t):
        h = self.ent_embeddings(h)
        r = self.Vr(r)
        t = self.ent_embeddings(t)
        score = self._calc(h, r, t)
        return score

class adv_DistMult(nn.Module):
    def __init__(self, ent_tot, rel_tot, em_dim = 100, dropout=False, dropout_p = 0.0):
        super(adv_DistMult, self).__init__()
        self.ent_embeddings = nn.Embedding(ent_tot, em_dim, padding_idx=0)
        self.Vr = nn.Embedding(rel_tot, em_dim, padding_idx=0)
        #self.W = nn.Parameter(torch.zeros((em_dim, em_dim)), requires_grad=True)
        self.W = nn.Parameter(torch.from_numpy(np.eye(em_dim)), requires_grad=False)
        self.init_weights()
        if dropout:
            self.dropout = nn.Dropout(dropout_p)
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.Vr.weight.data)
        #nn.init.xavier_uniform_(self.W)
    def _calc(self, batch_h, batch_r, batch_t):
        # yh = torch.sigmoid(torch.mm(self.W, batch_h.T).T)
        # yt = torch.sigmoid(torch.mm(self.W, batch_t.T).T)
        #yh = torch.tanh(torch.mm(self.W, batch_h.T).T)
        #yt = torch.tanh(torch.mm(self.W, batch_t.T).T)

        #yh = torch.mm(self.W, batch_h.T).T
        #yt = torch.mm(self.W, batch_t.T).T
        #yh = batch_h
        #yt = batch_t
        yh = torch.sigmoid(batch_h)
        yt = torch.sigmoid(batch_t)


        # yh = torch.mm(self.W, batch_h.T).T
        # yt = torch.mm(self.W, batch_t.T).T
        # print(yh.shape, yt.shape, batch_r.shape)
        hidden = torch.sum(yh * batch_r * yt, -1)
        return 
    def forward(self, batch_h, batch_r, batch_t, batch_size):
        batch_h = self.ent_embeddings(batch_h)
        batch_r = self.Vr(batch_r)
        batch_t = self.ent_embeddings(batch_t)
        score = self._calc(batch_h, batch_r, batch_t)
        pos_score = score[:batch_size]
        neg_score = score[batch_size:]
        return pos_score, neg_score
    def predict(self, h, r, t):
        h = self.ent_embeddings(h)
        r = self.Vr(r)
        t = self.ent_embeddings(t)
        score = self._calc(h, r, t)
        return score
'''

if __name__=='__main__':
    model = DistMult(10, 10, 3)
    # print(model.W)
    # print(model.ent_embeddings(torch.LongTensor([1])))
    h = torch.LongTensor([1, 2, 3])
    r = torch.LongTensor([1, 3, 3])
    t = torch.LongTensor([1, 2, 3])
    a, b = model(h, r, t, 2)
    # print(a.shape, b.shape)
    # a = model(h, r, t)








