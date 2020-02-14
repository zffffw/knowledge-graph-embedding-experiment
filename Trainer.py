import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import codecs
class Trainer:
    def __init__(self, params, model_name, train_data_loader, valid_data_loader, model, \
        loss, optimizer, batch_size, negative_size, times=100, use_GPU=True, check_step = 10, save_step = 20):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data_loader = train_data_loader.to(self.device)
        self.valid_data_loader = valid_data_loader.to(self.device)
        self.best_valid_loss = ''
        self.check_step = check_step
        self.save_root = 'checkpoint/'
        self.losses = []
        self.times = times
        self.loss = loss.to(self.device)
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.params = params
        self.negative_size = negative_size
        self.batch_size = batch_size
        self.save_step = save_step
        self.save_best_name = self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim)\
                 +'.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.loss_' + self.params.loss + '.best.ckpt'
    
    def train_one_step(self, h, r, t):
        self.optimizer.zero_grad()
        # print(h.shape, r.shape, t.shape)
        p_score, n_score = self.model(h, r, t, self.batch_size)
        # p_score, n_score = p_score.to(self.device), n_score.to(self.device)
        # print(p_score.shape, n_score.shape)
        loss_ = self.loss(p_score, n_score, torch.Tensor([-1]).to(self.device)).to(self.device)
        loss_.backward()
        self.optimizer.step()
        return loss_.item()



        
    def save_check_point(self, epochs, isBest=False):
        if isBest:
            torch.save(self.model.state_dict(),   self.save_best_name)
        else:
            torch.save(self.model.state_dict(),   self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim)\
                 +'.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.loss_' + self.params.loss + '.epoch_' + str(epochs) + '.ckpt')
        
    
    def valid_model(self, epochs, train_loss):
        ts = int(time.time())
        self.model.eval()
        valid_loss = 0
        fw1 = codecs.open(self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim) +\
            '.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.check.txt', 'a+', encoding='utf-8')
        
        for data_val in self.valid_data_loader:
            batch_h, batch_r, batch_t, h_n, r_n, t_n = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n']
            for i in range(len(h_n)):
                batch_h = torch.cat((batch_h, h_n[i]), 0)
                batch_t = torch.cat((batch_t, t_n[i]), 0)
                batch_r = torch.cat((batch_r, r_n[i]), 0)
            # batch_h, batch_t, batch_r = h.to(self.device), t.to(self.device), r.to(self.device)
            p_score, n_score = self.model(batch_h, batch_r, batch_t, self.batch_size)
            loss_ = self.loss(p_score, n_score, torch.Tensor([-1]).to(self.device))
            valid_loss += loss_.item()
        fw1.write('epoch:{}, valid loss:{:.2f}, train loss:{:.2f}, timestep:{}\n'.format(epochs, valid_loss, train_loss, ts))
        if self.best_valid_loss == '' or self.best_valid_loss > valid_loss:
            self.best_valid_loss = valid_loss
            fw2 = codecs.open(self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim) +\
            '.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.bset_check.txt', 'a+', encoding='utf-8')
            fw2.write('epoch:{}, valid loss:{:.2f}, train loss:{:.2f}, timestep:{}\n'.format(epochs, valid_loss, train_loss, ts))
            self.save_check_point(epochs, isBest=True)
    
    def run(self):
        for epoch in range(self.times):
            cur_loss = 0
            for n, data_val in enumerate(self.train_data_loader):
                h, r, t, h_n, r_n, t_n = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n']
                for i in range(len(h_n)):
                    h = torch.cat((h, h_n[i]), 0)
                    t = torch.cat((t, t_n[i]), 0)
                    r = torch.cat((r, r_n[i]), 0)
                batch_h, batch_t, batch_r = h.to(self.device), t.to(self.device), r.to(self.device)
                cur_loss += self.train_one_step(batch_h, batch_r, batch_t)
            
                print('{}/{}, {:.2%}'.format(n, len(self.train_data_loader), n/len(self.train_data_loader)), end='\r')
            self.losses.append(cur_loss)
            if (epoch) % self.check_step == 0:
                self.valid_model(epoch, cur_loss)
                print('[epochs:{}, cur_loss:{:.2f}]'.format(epoch + 1, self.losses[-1]))
            if (epoch) % self.save_step == 0:
                self.save_check_point(epoch)
        


    