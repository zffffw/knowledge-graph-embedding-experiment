import torch
from torch.utils.data import Dataset, DataLoader
import codecs
import numpy as np
import random


class kge_data_loader(Dataset):
    def __init__(self, root_dir, file_name, ent_tot, sample_flag=True, sample_size=1):
        self.root_dir = root_dir
        self.file_name = file_name
        fr =codecs.open(root_dir + '/' + file_name, 'r', encoding='utf-8')
        self.ent_tot = ent_tot
        self.data_frame = [line.strip().split('\t') for line in fr.readlines()[1:]]
        self.sample_flag = sample_flag
        self.sample_size = sample_size
        
        # print(len(self.data_frame))
    def create_label(self):
        print('[creating labels]')
        for i in range(len(self.data_frame)):
            label = eval(self.data_frame[i][3])
            one_hot = list(torch.zeros(self.ent_tot).scatter_(0, torch.LongTensor(label), 1))
            # one_hot = ((1.0 - self.label_smoothing)*one_hot) + (1.0/one_hot.size(0))
            self.data_frame[i][3] = one_hot
            print('{:3f} {}/{}'.format(i/len(self.data_frame), i, len(self.data_frame)), end='\r')
        print('[create labels ok!]')
        # print(self.data_frame)




    def __len__(self):
        return len(self.data_frame)
    def sample_neg(self, idx):
        h = int(self.data_frame[idx][0])
        r = int(self.data_frame[idx][1])
        t = int(self.data_frame[idx][2])
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
        r = int(self.data_frame[idx][1])
        h, r, t, h_n, r_n, t_n = self.sample_neg(idx)
        try:
            return {'en1':h, 'en2':t, 'rel':r, 'en1_n':h_n, 'en2_n':t_n, 'rel_n':r_n, 'en1_neighbour':self.data_frame[idx][3]}
        except:
            return {'en1':h, 'en2':t, 'rel':r, 'en1_n':h_n, 'en2_n':t_n, 'rel_n':r_n, 'en1_neighbour':[]}
        




if __name__=='__main__':
    train_loader = kge_data_loader('toy', 'train.txt', ent_tot=14000, sample_flag = True, sample_size=0)
    dataset_loader = DataLoader(train_loader, batch_size=5, shuffle=False)
    k = 0
    for data_val in dataset_loader:
        h, r, t, h_n, r_n, t_n = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n']
        # k += 8
        print(h, r, t, h_n, r_n, t_n)
        print(data_val['en1_neighbour'])
        break
        
        # for i in range(len(h_n)):
        #     h = torch.cat((h, h_n[i]), 0)
        #     t = torch.cat((t, t_n[i]), 0)
        #     r = torch.cat((r, r_n[i]), 0)

        # print(k, h.shape,  end='\r')
    
        



    