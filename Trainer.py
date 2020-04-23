import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import codecs
import numpy as np
from Tester import Tester


class Trainer:
    def __init__(self, params, ent_tot, rel_tot, train_data_loader, valid_data_loader, model, optimizer):
        self.model_name = params.model
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.best_valid_loss = ''
        self.check_step = params.check_step
        self.eval_step = params.eval_step
        self.save_root = 'checkpoint/' + params.model + '/' + params.data + '/'
        self.losses = []
        self.eval_mrr_filter = 0.0
        self.eval_h = {1:0.0, 3:0.0, 10:0.0}
        self.times = params.times
        self.loss_name = params.loss
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.params = params
        self.negative_size = params.negative_sample_size
        self.batch_size = params.batch_size
        self.save_step = params.save_step
        self.early_stop = 0

        self.save_best_name = self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim)\
                 +'.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' +  \
                     self.params.optimizer + '.loss_' + self.params.loss +'.batch_size_' +  \
                     str(self.params.batch_size) 
        if self.params.cluster_ent_name:
            self.save_best_name +=  '.cluster_ent_' + self.params.cluster_ent_name 
        if self.params.cluster_ent_name2:
            self.save_best_name +=  '.cluster_ent2_' + self.params.cluster_ent_name2 
        if self.params.cluster_rel_name:
            self.save_best_name +=  '.cluster_rel_' + self.params.cluster_rel_name
        if self.params.cluster_rel_name2:
            self.save_best_name +=  '.cluster_rel2_' + self.params.cluster_rel_name2
        
        self.save_best_name += '.best.ckpt'
        self.fw_log = codecs.open(self.save_best_name + '.txt', 'w', encoding='utf-8')
        

        

    def calc_loss(self, h, r, t, p_score, n_score, size, label=[]):
        if self.loss_name == 'margin':
            n_score = n_score.reshape(-1, self.negative_size)
            p_score = p_score.reshape(-1, 1)
            closs = self.model.loss(p_score, n_score, torch.Tensor([-1]).to(self.device))
        elif self.loss_name == 'bce':
            if self.params.mode == 'kvsall':
                closs = self.model.loss(p_score, label)
            elif self.params.mode == 'neg_sample':
                label = torch.cat((torch.ones(p_score.shape[0]), torch.zeros(n_score.shape[0]))).to(self.device)
                
                score = torch.cat((p_score, n_score), -1).flatten().to(self.device)
                # print(label.shape,score.shape)
                # exit(0)
                # print(p_score.shape, n_score.shape)
                closs = self.model.loss(score, label)
            elif self.params.mode ==  '1vsall':
                label = label.to(self.device)
                closs = self.model.loss(p_score, label)
        elif self.loss_name == 'ce':
            label = label.to(self.device)
            closs = self.model.loss(p_score, label)
        elif self.loss_name == 'sfmargin':
            # print('----------------------')
            # print(sum(p_score))
            # print(sum(n_score))
            # n_score = n_score.reshape(-1, self.negative_size)
            # p_score = p_score.reshape(-1, 1)
            score = torch.cat((p_score, n_score), -1).flatten().to(self.device)
            
            label = torch.cat((torch.ones(p_score.shape[0]), -torch.ones(n_score.shape[0]))).to(self.device)
            # print(score, label)
            # time.sleep(2)
            closs = self.model.loss(score, label)
        if self.params.regularize != 0.0:
            closs += self.model.regul(h, r, t)
        return closs

            
    def train_one_step(self, h, r, t, label=[]):
        self.optimizer.zero_grad()
        size = int(h.shape[0] / (1 + self.negative_size))
        p_score, n_score = self.model(h, r, t, size)
        # print(p_score.shape, n_score.shape)
        loss_ = self.calc_loss(h, r, t, p_score, n_score, size, label).to(self.device)
        # if loss_.item() == np.nan:
        
        # print(loss_.item())
        
        loss_.backward()
        if self.model_name == 'ComplEx' or self.model_name == 'DistMult':
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
        self.optimizer.step()
        # print(self.model.grad)
        return loss_.item()



        
    def save_check_point(self, epochs):
        print('saving model...')
        torch.save(self.model.state_dict(),   self.save_best_name)
    
    def eval_model(self, epochs):
        eval_ = Tester(self.params, self.ent_tot, self.rel_tot, self.model, self.valid_data_loader)
        # raw_mrr1, filter_mrr1, Hist_raw_n, Hist_filter_n = eval_.test_run(ttype='head')
        filter_mrr, Hist_filter_n = eval_.test_run(mode='eval')
        print("#####total#####")
        print("MRR:{:.6f}, Hist@1:{:.3f}, Hist@3:{:.3f}, Hist@10:{:.3f}".format(filter_mrr, \
                                    np.mean(Hist_filter_n[0]), np.mean(Hist_filter_n[2]), np.mean(Hist_filter_n[9])))
        

        self.fw_log.write('\n###eval tail###\n')
        self.fw_log.write('MRR:{:.6f}\n'.format(filter_mrr))
        for i in [1, 3, 10]:
            self.fw_log.write("# filter Hist@{} : {:.3f}\n".format(i, np.mean(Hist_filter_n[i - 1])))
        if filter_mrr > self.eval_mrr_filter:
            # print(abs(filter_mrr - self.eval_mrr_filter), np.mean(Hist_filter_n[10 - 1]), 1, '\n')
            self.eval_mrr_filter = filter_mrr
            self.eval_h[10] = np.mean(Hist_filter_n[10 - 1])
            self.save_check_point(epochs)
            # if self.model_name == 'TransE':
            #     self.model.save_embeddings()
            self.early_stop = 0
        elif abs(filter_mrr - self.eval_mrr_filter) < 1e-4 and np.mean(Hist_filter_n[10 - 1]) > self.eval_h[10]:
            # print(abs(filter_mrr - self.eval_mrr_filter), np.mean(Hist_filter_n[10 - 1]), 2, '\n')
            self.eval_h[10] = np.mean(Hist_filter_n[10 - 1])
            self.eval_mrr_filter = filter_mrr
            self.save_check_point(epochs)
            # if self.model_name == 'TransE':
            #     self.model.save_embeddings()
            self.early_stop = 0
        else:
            if filter_mrr < 0.005:
                self.early_stop += 100
            self.early_stop += 1




        

    
    def run(self):
        
        for epoch in range(self.times):
            self.model.train()
            self.model.to(self.device)
            cur_loss = 0
            tot = 0
            for n, data_val in enumerate(self.train_data_loader):
                h, r, t, h_n, r_n, t_n, label = data_val['h'], data_val['rel'], data_val['t'], data_val['h_n'], data_val['rel_n'],data_val['t_n'], data_val['t_multi_1']
                tot += h.shape[0]
                cbatch = h.shape[0]
                for i in range(len(h_n)):
                    h = torch.cat((h, h_n[i]), 0)
                    t = torch.cat((t, t_n[i]), 0)
                    r = torch.cat((r, r_n[i]), 0)
                batch_h, batch_t, batch_r = h.to(self.device), t.to(self.device), r.to(self.device)
                if self.params.loss == 'bce' and self.params.mode == 'kvsall':
                    label = label.to(self.device)
                # print(label.shape)  
                cur_loss += self.train_one_step(batch_h, batch_r, batch_t, label)
                
                # print('epochs:{}, {}/{}, {:.2%}                      '.format(epoch + 1, n, len(self.train_data_loader), n/len(self.train_data_loader)), end='\r')
            print('epochs:{}, loss:{:.6f}                '.format(epoch + 1, cur_loss), end='\r')
            
            self.losses.append(cur_loss)
            # if (epoch + 1) % self.check_step == 0:
            #     self.valid_model(epoch, cur_loss)
            #     print('[epochs:{}, cur_loss:{:.2f}]'.format(epoch + 1, self.losses[-1]))
            # if (epoch + 1) % self.save_step == 0:
            #     self.save_check_point(epoch)
            if (epoch + 1) % self.eval_step == 0:
                print('epochs:{}, loss:{:.6f}                '.format(epoch + 1, cur_loss))
                self.fw_log.write('\nepochs:{}, loss:{:.6f}                \n'.format(epoch + 1, cur_loss))
                mrr = self.eval_model(epoch)
                stop_num = 4
                if self.early_stop >= stop_num:
                    print('eval metric descending over {} period, early stop!!'.format(stop_num))
                    self.fw_log.write('eval metric descending over {} period, early stop!!\n'.format(stop_num))
                    break
        print("\n{:.3f}, {:.4f}".format(self.eval_mrr_filter, self.eval_h[10]))
        self.fw_log.close()
        


    
