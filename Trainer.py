import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import codecs

class Trainer:
    def __init__(self, params, ent_tot, rel_tot, model_name, loss_name, train_data_loader, valid_data_loader, model, \
        loss, optimizer, batch_size, negative_size, times=100, use_GPU=True, check_step = 10, save_step = 20):
        self.model_name = model_name
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.best_valid_loss = ''
        self.check_step = check_step
        self.save_root = 'checkpoint/'
        self.losses = []

        self.times = times
        self.loss = loss.to(self.device)
        self.loss_name = loss_name
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.params = params
        self.negative_size = negative_size
        self.batch_size = batch_size
        self.save_step = save_step
        self.save_best_name = self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim)\
                 +'.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.loss_' + self.params.loss + '.best.ckpt'

    def calc_loss(self, t, p_score, n_score, size, label=[]):
        # print(p_score.shape, n_score.shape, t.shape)
        if self.loss_name == 'margin':
            p_t = t[:size].reshape(-1, 1)
            n_t = t[size:].reshape(-1, 1)
            # print(n_t, p_t)
            p_score_s = torch.gather(p_score, dim=1, index=p_t).to(self.device)
            n_score_s = torch.gather(n_score, dim=1, index=n_t).to(self.device)
            # print(p_score_s.shape, n_score_s.shape)
            closs = self.loss(p_score_s, n_score_s, torch.Tensor([-1]).to(self.device)).to(self.device)
        elif self.loss_name == 'bce':
            # print(p_score.shape, label.shape)
            closs = self.loss(p_score, label).to(self.device)
        return closs

            
    def train_one_step(self, h, r, t, label=[]):
        self.optimizer.zero_grad()
        # print(h.shape, r.shape, t.shape)
        size = int(h.shape[0] / (1 + self.negative_size))
        # print(size)
        p_score, n_score = self.model(h, r, t, size)
        # print(p_score.shape, n_score.shape)
        # print(t.shape, h.shape)
        # p_score, n_score = p_score.to(self.device), n_score.to(self.device)
        # print(p_score.shape, n_score.shape)
        # print(label)
        loss_ = self.calc_loss(t, p_score, n_score, size, label)
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
            h, r, t, h_n, r_n, t_n, label = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n'], data_val['en1_neighbour']
            for i in range(len(h_n)):
                h = torch.cat((h, h_n[i]), 0)
                t = torch.cat((t, t_n[i]), 0)
                r = torch.cat((r, r_n[i]), 0)
            # label = self.label_transform(label).to(self.device)
            if self.params.loss == 'bce':
                label = self.label_transform(label).to(self.device)
            batch_h, batch_t, batch_r = h.to(self.device), t.to(self.device), r.to(self.device)
            size = int(h.shape[0] / (1 + self.negative_size))
            p_score, n_score = self.model(batch_h, batch_r, batch_t, size)

            loss_ = self.calc_loss(batch_t, p_score, n_score, size, label)
            valid_loss += loss_.item()
        fw1.write('epoch:{}, valid loss:{:.2f}, train loss:{:.2f}, timestep:{}\n'.format(epochs, valid_loss, train_loss, ts))
        if self.best_valid_loss == '' or self.best_valid_loss > valid_loss:
            self.best_valid_loss = valid_loss
            fw2 = codecs.open(self.save_root + self.model_name + '.emb_' +  str(self.params.embedding_dim) +\
            '.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' + self.params.optimizer + '.bset_check.txt', 'a+', encoding='utf-8')
            fw2.write('epoch:{}, valid loss:{:.2f}, train loss:{:.2f}, timestep:{}\n'.format(epochs, valid_loss, train_loss, ts))
            self.save_check_point(epochs, isBest=True)
    
    # change str label to torch
    def label_transform(self,  label):
        res = 0
        flag = False
        # print(label)
        for i in label:
            tmp = eval(i)
            one_hot = torch.zeros(self.ent_tot).scatter_(0, torch.LongTensor(tmp), 1)
            #label smoothing
            # e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))
            one_hot = ((1.0 - self.params.label_smoothing)*one_hot) + (1.0/one_hot.size(0))
            # print(one_hot)
            if flag:
                res = torch.cat((res, one_hot), -1)
            else:
                flag = True
                res = one_hot
        
                
        return res.reshape(len(label), -1)
    
    def run(self):
        for epoch in range(self.times):
            cur_loss = 0
            self.model.train()
            for n, data_val in enumerate(self.train_data_loader):
                h, r, t, h_n, r_n, t_n, label = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n'], data_val['en1_neighbour']
                for i in range(len(h_n)):
                    h = torch.cat((h, h_n[i]), 0)
                    t = torch.cat((t, t_n[i]), 0)
                    r = torch.cat((r, r_n[i]), 0)
                # print(label)
                if self.params.loss == 'bce':
                    label = self.label_transform(label).to(self.device)
                # label = label.to(self.device)
                # print(label)
                batch_h, batch_t, batch_r = h.to(self.device), t.to(self.device), r.to(self.device)
                cur_loss += self.train_one_step(batch_h, batch_r, batch_t, label)
            
                print('{}/{}, {:.2%}'.format(n, len(self.train_data_loader), n/len(self.train_data_loader)), end='\r')
            self.losses.append(cur_loss)
            if (epoch) % self.check_step == 0:
                self.valid_model(epoch, cur_loss)
                print('[epochs:{}, cur_loss:{:.2f}]'.format(epoch + 1, self.losses[-1]))
            if (epoch) % self.save_step == 0:
                self.save_check_point(epoch)
        


    