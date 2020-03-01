import torch
import torch.nn as nn
import torch.nn.functional as F
import codecs
import numpy as np
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
        self.params = params
    
    def test_run(self, type='head', hist=[1, 3, 10]):
        self.model.eval()
        raw_mrr = 0.0
        filter_mrr = 0.0
        Hist_raw_n = [0.0 for i in range(10)]
        Hist_filter_n = [0.0 for i in range(10)]
        tot = 0
        for n, data_val in enumerate(self.test_data_loader):
            #only use h, r, t, label
            h, r, t, h_n, r_n, t_n, label = data_val['h'], data_val['rel'], data_val['t'], data_val['h_n'], data_val['rel_n'],data_val['t_n'], data_val['h_neighbour_1']
            h, r = h.to(self.device), r.to(self.device)
            tot += int(h.shape[0]) # tot test size
            all_tail_score_raw = self.model.predict(h, r, t)
            print(all_tail_score_raw.shape)
            all_tail_score_filter = all_tail_score_raw.clone()
            label_ = []
            for i in label:
                label_.append(eval(i))
            for i in range(t.shape[0]):
                target = all_tail_score_filter[i][t[i].item()].item()
                if self.params.loss in ['margin']:
                    all_tail_score_filter[i][label_[i]] = 10000
                elif self.params.loss in ['bce', 'ce']:
                    all_tail_score_filter[i][label_[i]] = 0.0
                all_tail_score_filter[i][t[i]] = target
            descending = True if self.params.loss in ['bce', 'ce'] else False
            sorted_data_raw, indices_raw = torch.sort(all_tail_score_raw, -1, descending=descending)
            sorted_data_filter, indices_filter = torch.sort(all_tail_score_filter, -1, descending=descending)
            # print(sorted_data_filter.shape)
            for i in range(t.shape[0]):
                raw_rank = np.argwhere(indices_raw[i] == t[i].item())[0][0] + 1.0
                filter_rank = np.argwhere(indices_filter[i] == t[i].item())[0][0] + 1.0
                raw_mrr += 1.0 / (float(raw_rank))
                filter_mrr += 1.0 / (float(filter_rank))
                for k in range(10):
                    if filter_rank <= k + 1:
                        Hist_filter_n[k] += 1.0
                    if raw_rank <= k + 1:
                        Hist_raw_n[k] += 1.0
            print('{}, {:.3f}'.format(tot, raw_mrr / ((n + 1)*100)), end='\r')

        print(tot)
        print("# raw MRR:{:.8f}".format(raw_mrr / tot))
        print("# filter MRR:{:.8f}".format(filter_mrr / tot))
        for i in hist:
            print("# raw Hist@{} : {:.3f}".format(i, Hist_raw_n[i - 1] / tot))
        for i in hist:
            print("# filter Hist@{} : {:.3f}".format(i, Hist_filter_n[i - 1] / tot))
        return raw_mrr, filter_mrr, Hist_raw_n, Hist_filter_n
            

                

    




if __name__=='__main__':
    get_triples_from_all_datasets('data/FB15k')