import torch
import torch.nn as nn
import torch.nn.functional as F
import codecs
from config import *

def get_triples_from_all_datasets(root):
    names = ['train2index.txt', 'test2index.txt', 'valid2index.txt']
    tail_predict = {}
    head_predict = {}
    for name in names:
        fr = codecs.open(root + '/' + name, 'r', encoding='utf-8')
        for line in fr.readlines()[1:]:
            h, r, t = line.strip().split('\t')
            h, r, t = int(h), int(r), int(t)
            if (h, r) not in tail_predict:
                tail_predict[(h, r)] = []
            if (r, t) not in head_predict:
                head_predict[(r, t)] = []
            tail_predict[(h, r)].append(t)
            head_predict[(r, t)].append(h)
    return tail_predict, head_predict


class Tester(object):
    def __init__(self, params, ent_tot, rel_tot, model, test_data_loader):
        self.device = 'cpu'
        self.model = model.to(self.device)
        self.test_data_loader = test_data_loader
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.head_predict, self.tail_predict = get_triples_from_all_datasets(datasets_param.d[params.data]['root'])

    
    def test_one_step(self, test_data):
        res = self.model.predict(test_data[:, 0], test_data[:, 1], test_data[:, 2])
        # return torch.cat((test_data.float(), res), 1)
        return res
    
    def get_rank(self, res, target, type='head'):
        raw_rank = 0
        fil_rank = 0
        c = 0
        for n, i in enumerate(res):
            if type == 'head':
                if int(i[0]) == target:
                    raw_rank = n + 1
                    fil_rank = n + 1 - c
                    break
                if int(i[0]) in self.tail_predict[(int(i[1]), int(i[2]))] and not fil_rank:
                    c += 1
                    # fil_rank = n + 1
            elif type == 'tail':
                if int(i[2]) == target:
                    raw_rank = n + 1
                    fil_rank = n + 1 - c
                    break
                if int(i[2]) in self.head_predict[(int(i[0]), int(i[1]))] and not fil_rank:
                    # fil_rank = n + 1
                    c += 1
        return raw_rank, fil_rank
        

    def test_run(self, type='head'):
        self.model.eval()
        tot_rank = 0
        tot_rank_reverse = 0
        Hist_10 = 0
        Hist_3 = 0
        Hist_1 = 0
        fil_tot_rank = 0
        fil_tot_rank_reverse = 0
        fil_Hist_10 = 0
        fil_Hist_3 = 0
        fil_Hist_1 = 0
        
        for n, data_val in enumerate(self.test_data_loader):
            h, r, t = data_val['en1'], data_val['rel'], data_val['en2']
            if type == 'head':
                # test_data = torch.cat((torch.arange(0, self.ent_tot).reshape(-1, 1), torch.Tensor([r]*self.ent_tot).long().reshape(-1, 1),\
                #  torch.Tensor([t]*self.ent_tot).long().reshape(-1, 1)), 1)
                print(h, r, t)
                tmp = self.test_one_step(test_data.long().to(self.device)).detach().numpy()
                tmp = sorted(tmp, key=lambda x: x[3])
                raw_rank, fil_rank = self.get_rank(tmp, h, type)

                
            elif type == 'tail':
                # test_data[:, 2] = torch.arange(0, self.ent_tot)
                # test_data[:, 0] = h
                test_data = torch.cat((torch.Tensor([h]*self.ent_tot).long().reshape(-1, 1) , torch.Tensor([r]*self.ent_tot).long().reshape(-1, 1),\
                 torch.arange(0, self.ent_tot).reshape(-1, 1)), 1)
                tmp = self.test_one_step(test_data.long().to(self.device)).detach().numpy()
                tmp = sorted(tmp, key=lambda x: x[3])
                raw_rank, fil_rank = self.get_rank(tmp, t, type)
            print('mr:{:.3f}, mrr:{:.3f}, raw_Hist@10:{:.2%}, raw_Hist@3:{:.2%}, raw_Hist@1:{:.2%}, mr:{:.3f}, mrr:{:.3f}, filter_Hist@10:{:.2%}, filter_Hist@3:{:.2%}, filter_Hist@1:{:.2%}, {}/{}, {:.2%}'.\
                     format(tot_rank / (n + 1), tot_rank_reverse / (n + 1), Hist_10 / (n + 1), Hist_3 / (n + 1), Hist_1 / (n + 1), \
                         fil_tot_rank / (n + 1), fil_tot_rank_reverse / (n + 1),  fil_Hist_10 / (n + 1), fil_Hist_3 / (n + 1), fil_Hist_1 / (n + 1),\
                          n, len(self.test_data_loader), n/len(self.test_data_loader)), end='\r')
            tot_rank += raw_rank
            tot_rank_reverse += 1.0/raw_rank
            fil_tot_rank += fil_rank
            fil_tot_rank_reverse += 1.0/fil_rank
            if raw_rank <= 10:
                Hist_10 += 1
            if raw_rank <= 3:
                Hist_3 += 1
            if raw_rank <= 1:
                Hist_1 += 1
            if fil_rank <= 10:
                fil_Hist_10 += 1
            if fil_rank <= 3:
                fil_Hist_3 += 1
            if fil_rank <= 1:
                fil_Hist_1 += 1

            
        n += 1
        mean_rank = tot_rank / n
        mean_rank_reverse = tot_rank_reverse / n
        Hist_10 /= n
        Hist_3 /= n
        Hist_1 /= n
        fil_mean_rank = fil_tot_rank / n
        fil_mean_rank_reverse = fil_tot_rank_reverse / n
        fil_Hist_10 /= n
        fil_Hist_3 /= n
        fil_Hist_1 /= n
        print('\n', type, 'raw_mr:{:.3f}, raw_mrr:{:.3f}, raw_Hist@10:{:.2%}, raw_Hist@3:{:.2%}, raw_Hist@1:{:.2%},\
             fil_mr:{:.3f}, fil_mrr:{:.3f},  fil_Hist@10:{:.2%}, fil_Hist@3:{:.2%}, fil_Hist@1:{:.2%}'.
             format(mean_rank, mean_rank_reverse, Hist_10, Hist_3, Hist_1, fil_mean_rank, fil_mean_rank_reverse, fil_Hist_10, fil_Hist_3, fil_Hist_1))





if __name__=='__main__':
    get_triples_from_all_datasets('data/FB15k')