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
        self.test_batch_size = params.test_batch_size
    
    def replace_all_entities(self, h, r, t, rtype = 'head'):
        if rtype == 'head':
            h = torch.arange(0, self.ent_tot).reshape(1, -1).repeat(r.shape[0], 1)
            r = r.reshape(-1, 1).repeat(1, self.ent_tot)
            t = t.reshape(-1, 1).repeat(1, self.ent_tot)
        elif rtype == 'tail':
            t = torch.arange(0, self.ent_tot).reshape(1, -1).repeat(r.shape[0], 1)
            r = r.reshape(-1, 1).repeat(1, self.ent_tot)
            h = h.reshape(-1, 1).repeat(1, self.ent_tot)
        return h, r, t

    def test_run(self, ttype='tail', hist=[1, 3, 10]):
        self.model.eval()
        raw_mrr = 0.0
        filter_mrr = 0.0
        Hist_raw_n = [[] for i in range(10)]
        Hist_filter_n = [[] for i in range(10)]
        tot = 0
        for n, data_val in enumerate(self.test_data_loader):
            #only use h, r, t, label
            h, r, t, tail_label, head_label = data_val['h'], data_val['rel'], data_val['t'], data_val['h_neighbour_1'], data_val['t_neighbour_1']
            # all_h, all_r, all_t = self.replace_all_entities(h, r, t, ttype)
            # all_h, all_r, all_t = all_h.to(self.device), all_r.to(self.device), all_t.to(self.device)
            # print(h, r, t)
            # print(all_h, all_r, all_t)
            # print(all_h.shape, all_r.shape, all_t.shape)
            cur_batch_size = h.shape[0]
            tot += int(cur_batch_size) # tot test size
            all_score_raw = self.model.predict(h, r, t)
            # for i in range(cur_batch_size):
            #     tmp = self.model.predict(all_h[i], all_r[i], all_t[i])
            #     if i == 0:
            #         all_score_raw = tmp
            #     else:
            #         all_score_raw = torch.cat((all_score_raw, tmp), -1)
            # print(all_score_raw)
            all_score_raw = all_score_raw.reshape(cur_batch_size, -1).cpu()
            # print(all_score_raw)
            all_score_filter = all_score_raw.clone()
            label_ = []
            
            # print(all_score_raw.shape, all_score_filter.shape)
            # print(target)
            for i in tail_label:
                label_.append(eval(i))
            for i in range(cur_batch_size):
                # if ttype == 'tail':
                #     target = t[i].item()
                # elif ttype == 'head':
                #     target = h[i].item()
                target = t[i].item()
                tmp = all_score_filter[i][target].item()
                if self.params.loss in ['margin']:
                    all_score_filter[i][label_[i]] = 10000
                elif self.params.loss in ['bce', 'ce']:
                    all_score_filter[i][label_[i]] = 0.0
                all_score_filter[i][target] = tmp
            descending = True if self.params.loss in ['bce', 'ce'] else False
            sorted_data_raw, indices_raw = torch.sort(all_score_raw, -1, descending=descending)
            sorted_data_filter, indices_filter = torch.sort(all_score_filter, -1, descending=descending)
            # print(sorted_data_filter.shape)
            for i in range(cur_batch_size):
                # if ttype == 'tail':
                #     target = t[i].item()
                # elif ttype == 'head':
                #     target = h[i].item()
                target = t[i].item()
                raw_rank = np.argwhere(indices_raw[i] == target)[0][0] + 1.0
                filter_rank = np.argwhere(indices_filter[i] == target)[0][0] + 1.0
                raw_mrr += 1.0 / (float(raw_rank))
                filter_mrr += 1.0 / (float(filter_rank))
                for k in range(10):
                    if filter_rank <= k + 1:
                        Hist_filter_n[k].append(1.0)
                    else:
                        Hist_filter_n[k].append(0.0)
                    if raw_rank <= k + 1:
                        Hist_raw_n[k].append(1.0)
                    else:
                        Hist_raw_n[k].append(0.0)
            print('{}, raw_mrr:{:.3f}, filter_mrr:{:.3f}'.format(tot, raw_mrr / tot, filter_mrr / tot), end='\r')
        print(tot)
        print("# raw MRR:{:.8f}".format(raw_mrr / tot))
        print("# filter MRR:{:.8f}".format(filter_mrr / tot))
        for i in hist:
            print("# raw Hist@{} : {:.3f}".format(i, np.mean(Hist_raw_n[i - 1])))
        for i in hist:
            print("# filter Hist@{} : {:.3f}".format(i, np.mean(Hist_filter_n[i - 1])))
        return raw_mrr, filter_mrr, Hist_raw_n, Hist_filter_n
            

                

    




if __name__=='__main__':
    get_triples_from_all_datasets('data/FB15k')