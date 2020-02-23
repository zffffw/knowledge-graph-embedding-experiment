import torch
from torch.utils.data import Dataset, DataLoader
import codecs
import numpy as np
import random
import pickle
import time

class kge_data_loader(Dataset):
    def __init__(self, root_dir, file_name, ent_tot, sample_flag=True, sample_size=1, params=None):
        self.root_dir = root_dir
        self.file_name = file_name
        # print(root_dir + '/' + file_name)
        fr =codecs.open(root_dir + '/' + file_name, 'rb')
        self.data_frame = pickle.load(fr)
        self.ent_tot = ent_tot
        # self.data_frame = [line.strip().split('\t') for line in fr.readlines()[1:]]
        self.sample_flag = sample_flag
        self.sample_size = sample_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.params = params
        # self.create_label()
        
        # print(len(self.data_frame))
    # def create_label(self):
    #     print('[creating labels]')
    #     for i in range(len(self.data_frame)//3):
    #         label = self.data_frame[i]['t_multi_1']
    #         # print(len(label))
    #         # print(max(label), self.ent_tot)
    #         # print(len(label))
    #         one_hot = torch.zeros(self.ent_tot).scatter_(0, torch.LongTensor(label), 1)
    #         # one_hot = ((1.0 - self.label_smoothing)*one_hot) + (1.0/one_hot.size(0))
    #         self.data_frame[i]['t_multi_1'] = one_hot
            
    #         # print(torch.sum(one_hot))
    #         print('{:3f} {}/{}'.format(i/len(self.data_frame), i, len(self.data_frame)), end='\r')
    #     print('[create labels ok!]')
        # print(self.data_frame)

    def label_transform(self, idx):
        label = self.data_frame[idx]['t_multi_1']
        one_hot = torch.zeros(self.ent_tot).scatter_(0, torch.LongTensor(label), 1)
        one_hot = ((1.0 - 0.1)*one_hot) + (1.0/one_hot.size(0))
        return one_hot




    def __len__(self):
        return len(self.data_frame)
    def sample_neg(self, idx):
        h = int(self.data_frame[idx]['h'])
        r = int(self.data_frame[idx]['r'])
        t = int(self.data_frame[idx]['t'])
        h_n = []
        t_n = []
        r_n = []
        for n in range(self.sample_size):
            if random.random() > 0.5:
                h_ = random.randint(0, self.ent_tot - 1)
                while h_ == h:
                    h_ = random.randint(0, self.ent_tot - 1)
                t_ = t
                
            else:
                t_ = random.randint(0, self.ent_tot - 1)
                while t_ == t:
                    t_ = random.randint(0, self.ent_tot - 1)
                h_ = h
            h_n.append(h_)
            t_n.append(t_)
            r_n.append(r)
        return h, r, t, h_n, r_n, t_n

    def __getitem__(self, idx):
        r = int(self.data_frame[idx]['r'])
        h, r, t, h_n, r_n, t_n = self.sample_neg(idx)
        try:
            if len(self.data_frame[idx]['t_multi_1']) == self.ent_tot:
                label = self.data_frame[idx]['t_multi_1']
            elif self.params.loss == 'margin':
                label = 0
            else:
                label = self.label_transform(idx)
            return {'en1':h, 'en2':t, 'rel':r, 'en1_n':h_n, 'en2_n':t_n, 'rel_n':r_n, 'en1_neighbour':label}
        except:
            return {'en1':h, 'en2':t, 'rel':r, 'en1_n':h_n, 'en2_n':t_n, 'rel_n':r_n, 'en1_neighbour':[]}
        




if __name__=='__main__':
    train_loader = kge_data_loader('FB15k', 'test.pkl', ent_tot=14951, sample_flag = True, sample_size=0)
    dataset_loader = DataLoader(train_loader, batch_size=100, shuffle=False, num_workers=6, pin_memory=True)
    k = 0
    start = time.time()
    for data_val in dataset_loader:
        h, r, t, h_n, r_n, t_n = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n']
        k += 1
        print(k,data_val['en1_neighbour'].shape, end='\r')
        # print(data_val['en1_neighbour'])
        # print(len(data_val['en1_neighbour']), data_val['en1_neighbour'][0].shape)
        # print(h, r, t, h_n, r_n, t_n)
        # print(data_val['en1_neighbour'])
        # break
        
        # for i in range(len(h_n)):
        #     h = torch.cat((h, h_n[i]), 0)
        #     t = torch.cat((t, t_n[i]), 0)
        #     r = torch.cat((r, r_n[i]), 0)

        # print(k, h.shape,  end='\r')
    end = time.time()
    print('\n', end - start)
    
        



    