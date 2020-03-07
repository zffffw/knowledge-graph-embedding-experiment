import torch
from torch.utils.data import Dataset, DataLoader
import codecs
import numpy as np
import random
import pickle
import time

class kge_data_loader(Dataset):
    def __init__(self, params, root_dir, file_name, ent_tot, mode):
        self.root_dir = root_dir
        self.file_name = file_name
        fr = codecs.open(root_dir + '/' + file_name, 'rb')
        self.data_frame = pickle.load(fr)
        self.flag_dict = {}
        self.ent_tot = ent_tot
        self.label_smoothing = params.label_smoothing
        self.negative_sample_size = params.negative_sample_size
        self.params = params
        self.mode = params.mode
        self.ttype = mode
        

        self.create_dict()
    def create_dict(self):
        for idx in self.data_frame:
            self.flag_dict[(self.data_frame[idx]['h'], self.data_frame[idx]['r'], self.data_frame[idx]['t'])] = 1

    def label_transform(self, idx):
        label = self.data_frame[idx]['t_multi_1']
        one_hot = torch.zeros(self.ent_tot)
        if self.mode == 'kvsall':
            onehot = one_hot.scatter_(0, torch.LongTensor(label), 1)
        elif self.mode == '1vsall':
            one_hot[self.data_frame[idx]['t']] = 1
        one_hot = ((1.0 - 0.1)*one_hot) + (1.0/one_hot.size(0))
        return one_hot




    def __len__(self):
        return len(self.data_frame)
    def sample_neg(self, idx):
        h = self.data_frame[idx]['h']
        r = self.data_frame[idx]['r']
        t = self.data_frame[idx]['t']
        h_n = []
        t_n = []
        r_n = []
        if self.ttype == 'test' or self.ttype == 'valid':
            return h, r, t, h_n, r_n, t_n
        for n in range(self.negative_sample_size):
            h_ = h
            r = r
            t_ = t
            while (h_, r, t_) in self.flag_dict:
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
            t_label = []
            h_label = []
            if self.ttype == 'train':
                if self.params.loss == 'bce':
                    t_label = self.label_transform(idx)
                elif self.params.loss == 'ce':
                    if self.mode == '1vsall':
                        t_label = self.data_frame[idx]['t']
            elif self.ttype == 'test' or self.ttype == 'valid':
                t_label = str(self.data_frame[idx]['t_multi_1'])
                h_label = str(self.data_frame[idx]['h_multi_1'])
            else:
                raise Exception("dataLoader Error: the mode of dataset must be train, valid or test")
            return {'h':h, 't':t, 'rel':r, 'h_n':h_n, 't_n':t_n, 'rel_n':r_n, 'h_neighbour_1':t_label, 't_neighbour_1':h_label}
        except Exception as e:
            raise Exception(e)
            
        




if __name__=='__main__':
    train_loader = kge_data_loader('toy', 'train.pkl', ent_tot=16, sample_flag = True, negative_sample_size=0, mode='train')
    dataset_loader = DataLoader(train_loader, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    k = 0
    print(train_loader.__getitem__(20))
    # start = time.time()
    # for data_val in dataset_loader:
    #     h, r, t, h_n, r_n, t_n = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n']
    #     k += 1
        # print(k,np.where(data_val['en1_neighbour'][0] == 1))
        # input()
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
    # end = time.time()
    # print('\n', end - start)
    
        



    