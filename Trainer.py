import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import codecs
from Tester import Tester

class Trainer:
    def __init__(self, params, ent_tot, rel_tot, train_data_loader, valid_data_loader, model, optimizer):
        self.model_name = params.model
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.device = torch.device("cuda:" + str(params.cuda) if params.cuda > -1 else "cpu")
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.best_valid_loss = ''
        self.check_step = params.check_step
        self.eval_step = params.eval_step
        self.save_root = 'checkpoint/' + params.model + '/' + params.data + '/'
        self.losses = []
        self.eval_mrr_filter = 0.0
        self.times = params.times
        self.loss_name = params.loss
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.params = params
        self.negative_size = params.negative_sample_size
        self.batch_size = params.batch_size
        self.save_step = params.save_step
        self.save_best_name = self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim)\
                 +'.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' +  \
                     self.params.optimizer + '.loss_' + self.params.loss +'.batch_size_' +  \
                     str(self.params.batch_size) + '.cn_' + str(self.params.entity_cluster_num) + '.best.ckpt'

    def calc_loss(self, t, p_score, n_score, size, label=[]):
        if self.loss_name == 'margin':
            n_score = n_score.reshape(-1, self.negative_size)
            p_score = p_score.reshape(-1, 1)
            closs = self.model.loss(p_score, n_score, torch.Tensor([-1]).to(self.device))
        elif self.loss_name == 'bce':
            if self.params.mode == 'kvsall':
                closs = self.model.loss(p_score, label)
            elif self.params.mode == 'neg_sample':
                # print(p_score.shape, n_score.shape)
                closs = self.model.loss(torch.cat((p_score, n_score), -1), label)
            elif self.params.mode ==  '1vsall':
                label = label.to(self.device)
                closs = self.model.loss(p_score, label)
        elif self.loss_name == 'ce':
            label = label.to(self.device)
            closs = self.model.loss(p_score, label)
        return closs

            
    def train_one_step(self, h, r, t, label=[]):
        self.optimizer.zero_grad()
        size = int(h.shape[0] / (1 + self.negative_size))
        p_score, n_score = self.model(h, r, t, size)
        # print(p_score.shape, n_score.shape)
        loss_ = self.calc_loss(t, p_score, n_score, size, label).to(self.device)
        loss_.backward()
        self.optimizer.step()
        return loss_.item()



        
    def save_check_point(self, epochs, isBest=False):
        if isBest:
            torch.save(self.model.state_dict(),   self.save_best_name)
            # self.model.save_embeddings(self.save_root)    
    
    # def valid_model(self, epochs, train_loss):
    #     ts = int(time.time())
    #     self.model.eval()
    #     valid_loss = 0
    #     fw1 = codecs.open(self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim) +\
    #         '.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.check.txt', 'a+', encoding='utf-8')
    #     for data_val in self.valid_data_loader:
    #         h, r, t, h_n, r_n, t_n, label = data_val['h'], data_val['rel'], data_val['t'], data_val['h_n'], data_val['rel_n'],data_val['t_n'], data_val['h_neighbour_1']
    #         for i in range(len(h_n)):
    #             h = torch.cat((h, h_n[i]), 0)
    #             t = torch.cat((t, t_n[i]), 0)
    #             r = torch.cat((r, r_n[i]), 0)
    #         batch_h, batch_t, batch_r = h.to(self.device, non_blocking=True), t.to(self.device, non_blocking=True), r.to(self.device, non_blocking=True)
    #         size = int(h.shape[0] / (1 + self.negative_size))
    #         if self.params.loss == 'bce':
    #             label = label.to(self.device, non_blocking=True)
    #         p_score, n_score = self.model(batch_h, batch_r, batch_t, size)
    #         loss_ = self.calc_loss(batch_t, p_score, n_score, size, label)
    #         valid_loss += loss_.item()
    #     fw1.write('epoch:{}, valid loss:{:.2f}, train loss:{:.2f}, timestep:{}\n'.format(epochs, valid_loss, train_loss, ts))
    #     if self.best_valid_loss == '' or self.best_valid_loss > valid_loss:
    #         self.best_valid_loss = valid_loss
    #         fw2 = codecs.open(self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim) +\
    #         '.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.bset_check.txt', 'a+', encoding='utf-8')
    #         fw2.write('epoch:{}, valid loss:{:.2f}, train loss:{:.2f}, timestep:{}\n'.format(epochs, valid_loss, train_loss, ts))
    #         self.save_check_point(epochs, isBest=True)
    
    def eval_model(self, epochs):
        eval_ = Tester(self.params, self.ent_tot, self.rel_tot, self.model, self.valid_data_loader)
        # raw_mrr1, filter_mrr1, Hist_raw_n, Hist_filter_n = eval_.test_run(ttype='head')
        raw_mrr, filter_mrr, Hist_raw_n, Hist_filter_n = eval_.test_run(ttype='tail')
        
        if filter_mrr > self.eval_mrr_filter:
            self.eval_mrr_filter = filter_mrr
            self.save_check_point(epochs, isBest=True)


        

    
    def run(self):
        for epoch in range(self.times):
            cur_loss = 0
            self.model.train()
            self.model.to(self.device)
            for n, data_val in enumerate(self.train_data_loader):
                h, r, t, h_n, r_n, t_n, label = data_val['h'], data_val['rel'], data_val['t'], data_val['h_n'], data_val['rel_n'],data_val['t_n'], data_val['h_neighbour_1']
                cbatch = h.shape[0]
                for i in range(len(h_n)):
                    h = torch.cat((h, h_n[i]), 0)
                    t = torch.cat((t, t_n[i]), 0)
                    r = torch.cat((r, r_n[i]), 0)
                
                batch_h, batch_t, batch_r = h.to(self.device, non_blocking=True), t.to(self.device, non_blocking=True), r.to(self.device, non_blocking=True)
                if self.params.loss == 'bce' and self.params.mode == 'kvsall':
                    label = label.to(self.device, non_blocking=True)
                elif self.params.loss == 'bce' and self.params.mode == 'neg_sample':
                    label = torch.zeros(h.shape[0]).scatter_(0, torch.arange(0, cbatch), 1).to(self.device)
                # print(label.shape)  
                cur_loss += self.train_one_step(batch_h, batch_r, batch_t, label)
                
                print('{}/{}, {:.2%}'.format(n, len(self.train_data_loader), n/len(self.train_data_loader)), end='\r')
            print('epochs:{}, loss:{:.6f}            '.format(epoch + 1, cur_loss / (epoch + 1.0)))
            self.losses.append(cur_loss)
            # if (epoch + 1) % self.check_step == 0:
            #     self.valid_model(epoch, cur_loss)
            #     print('[epochs:{}, cur_loss:{:.2f}]'.format(epoch + 1, self.losses[-1]))
            if (epoch + 1) % self.save_step == 0:
                self.save_check_point(epoch)
            if (epoch + 1) % self.eval_step == 0:
                mrr = self.eval_model(epoch)
                
        


    
