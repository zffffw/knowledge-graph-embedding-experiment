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
    
    # def test_one_step(self, test_data):
    #     res = self.model.predict(test_data[:, 0], test_data[:, 1], test_data[:, 2])
    #     # return torch.cat((test_data.float(), res), 1)
    #     return res
    def test_one_step(self, h, r):
        return self.model.predict(h, r)
    
    
    # def get_rank(self, res, target, type='head'):
    #     raw_rank = 0
    #     fil_rank = 0
    #     c = 0
    #     for n, i in enumerate(res):
    #         if type == 'head':
    #             if int(i[0]) == target:
    #                 raw_rank = n + 1
    #                 fil_rank = n + 1 - c
    #                 break
    #             if int(i[0]) in self.tail_predict[(int(i[1]), int(i[2]))] and not fil_rank:
    #                 c += 1
    #                 # fil_rank = n + 1
    #         elif type == 'tail':
    #             if int(i[2]) == target:
    #                 raw_rank = n + 1
    #                 fil_rank = n + 1 - c
    #                 break
    #             if int(i[2]) in self.head_predict[(int(i[0]), int(i[1]))] and not fil_rank:
    #                 # fil_rank = n + 1
    #                 c += 1
    #     return raw_rank, fil_rank
    def label_transform(self,  label):
        res = 0
        flag = False
        # print(label)
        for i in label:
            tmp = eval(i)
            one_hot =  torch.zeros(self.ent_tot).scatter_(0, torch.LongTensor(tmp), 1)
            # print(self.params.label_smoothing, one_hot.size(0))
            one_hot = ((1.0 - self.params.label_smoothing)*one_hot) + (float(self.params.label_smoothing)/one_hot.size(0))
            # print(one_hot)
            if flag:
                res = torch.cat((res, one_hot), -1)
            else:
                flag = True
                res = one_hot
                
        return res.reshape(len(label), -1)
    
    def test_run(self, type='head', hist=[1, 3, 10]):
        self.model.eval()
        raw_mrr = 0.0
        filter_mrr = 0.0
        Hist_raw_n = [0.0 for i in range(10)]
        Hist_filter_n = [0.0 for i in range(10)]
        tot = 0
        for n, data_val in enumerate(self.test_data_loader):
            print(tot, end='\r')
            #only use h, r, t, label
            h, r, t, h_n, r_n, t_n, label = data_val['en1'], data_val['rel'], data_val['en2'], data_val['en1_n'], data_val['rel_n'],data_val['en2_n'], data_val['en1_neighbour']
            h, r = h.to(self.device), r.to(self.device)
            tot += int(h.shape[0]) # tot test size
            tmp = self.test_one_step(h, r)
            # print(tmp[0])
            label_ = []
            for i in label:
                label_.append(eval(i))
            for i in range(t.shape[0]):
                # print(t[i])
                target = tmp[i][t[i].item()].item()
                tmp[i][label_[i]] = 0.0
                tmp[i][t[i]] = target
                # print(target)
            # label = self.label_transform(label).to(self.device)
            # label = label.to(self.device)
            # max_label_val = torch.max(label)
            # print(label)
            descending = True if self.params.loss == 'bce' else False
            # descending = False
            # print(descending)
            sorted_data, indices = torch.sort(tmp, -1, descending=descending)
            # print(sorted_data)
            # print(indices)
            # print(new_label)
            for i in range(t.shape[0]):
                # new_label = [label[i][j] for j in indices[i]]
                # print(label[i])
                # print(new_label)
                # raw_rank = np.argwhere(indices[i] == t[i])[0][0]
                filter_rank = np.argwhere(indices[i] == t[i].item())[0][0]
                # filter_rank = new_label.index(max_label_val)
                # print(raw_rank, filter_rank)
                # print(filter_rank)
                # raw_mrr += 1.0 / (float(raw_rank) + 1.0)
                filter_mrr += 1.0 / (float(filter_rank) + 1)
                # print(filter_rank, end='\r')
                # if raw_rank < 10:
                #     Hist_raw_n[raw_rank] += 1
                for k in range(10):
                    if filter_rank <= k:
                        Hist_filter_n[k] += 1.0

        print(tot)
        # print("# raw MRR:{:.3f}".format(raw_mrr / tot))
        print("# filter MRR:{:.3f}".format(filter_mrr / tot))
        # cur_raw_tot = 0
        # cur_filter_tot = 0
        # for i in hist:
        #     cur_raw_tot +=  Hist_raw_n[i - 1]
        #     print("# raw Hist@{} : {:.3f}".format(i, cur_raw_tot / tot))
        for i in hist:
            print("# filter Hist@{} : {:.3f}".format(i, Hist_filter_n[i - 1] / tot))

                

        

    # def test_run(self, type='head'):
    #     self.model.eval()
    #     tot_rank = 0
    #     tot_rank_reverse = 0
    #     Hist_10 = 0
    #     Hist_3 = 0
    #     Hist_1 = 0
    #     fil_tot_rank = 0
    #     fil_tot_rank_reverse = 0
    #     fil_Hist_10 = 0
    #     fil_Hist_3 = 0
    #     fil_Hist_1 = 0
        
    #     for n, data_val in enumerate(self.test_data_loader):
    #         h, r, t = data_val['en1'], data_val['rel'], data_val['en2']
    #         if type == 'head':
    #             # test_data = torch.cat((torch.arange(0, self.ent_tot).reshape(-1, 1), torch.Tensor([r]*self.ent_tot).long().reshape(-1, 1),\
    #             #  torch.Tensor([t]*self.ent_tot).long().reshape(-1, 1)), 1)
    #             print(h, r, t)
    #             tmp = self.test_one_step(test_data.long().to(self.device)).detach().numpy()
    #             tmp = sorted(tmp, key=lambda x: x[3])
    #             raw_rank, fil_rank = self.get_rank(tmp, h, type)

                
    #         elif type == 'tail':
    #             # test_data[:, 2] = torch.arange(0, self.ent_tot)
    #             # test_data[:, 0] = h
    #             test_data = torch.cat((torch.Tensor([h]*self.ent_tot).long().reshape(-1, 1) , torch.Tensor([r]*self.ent_tot).long().reshape(-1, 1),\
    #              torch.arange(0, self.ent_tot).reshape(-1, 1)), 1)
    #             tmp = self.test_one_step(test_data.long().to(self.device)).detach().numpy()
    #             tmp = sorted(tmp, key=lambda x: x[3])
    #             raw_rank, fil_rank = self.get_rank(tmp, t, type)
    #         print('mr:{:.3f}, mrr:{:.3f}, raw_Hist@10:{:.2%}, raw_Hist@3:{:.2%}, raw_Hist@1:{:.2%}, mr:{:.3f}, mrr:{:.3f}, filter_Hist@10:{:.2%}, filter_Hist@3:{:.2%}, filter_Hist@1:{:.2%}, {}/{}, {:.2%}'.\
    #                  format(tot_rank / (n + 1), tot_rank_reverse / (n + 1), Hist_10 / (n + 1), Hist_3 / (n + 1), Hist_1 / (n + 1), \
    #                      fil_tot_rank / (n + 1), fil_tot_rank_reverse / (n + 1),  fil_Hist_10 / (n + 1), fil_Hist_3 / (n + 1), fil_Hist_1 / (n + 1),\
    #                       n, len(self.test_data_loader), n/len(self.test_data_loader)), end='\r')
    #         tot_rank += raw_rank
    #         tot_rank_reverse += 1.0/raw_rank
    #         fil_tot_rank += fil_rank
    #         fil_tot_rank_reverse += 1.0/fil_rank
    #         if raw_rank <= 10:
    #             Hist_10 += 1
    #         if raw_rank <= 3:
    #             Hist_3 += 1
    #         if raw_rank <= 1:
    #             Hist_1 += 1
    #         if fil_rank <= 10:
    #             fil_Hist_10 += 1
    #         if fil_rank <= 3:
    #             fil_Hist_3 += 1
    #         if fil_rank <= 1:
    #             fil_Hist_1 += 1

            
    #     n += 1
    #     mean_rank = tot_rank / n
    #     mean_rank_reverse = tot_rank_reverse / n
    #     Hist_10 /= n
    #     Hist_3 /= n
    #     Hist_1 /= n
    #     fil_mean_rank = fil_tot_rank / n
    #     fil_mean_rank_reverse = fil_tot_rank_reverse / n
    #     fil_Hist_10 /= n
    #     fil_Hist_3 /= n
    #     fil_Hist_1 /= n
    #     print('\n', type, 'raw_mr:{:.3f}, raw_mrr:{:.3f}, raw_Hist@10:{:.2%}, raw_Hist@3:{:.2%}, raw_Hist@1:{:.2%},\
    #          fil_mr:{:.3f}, fil_mrr:{:.3f},  fil_Hist@10:{:.2%}, fil_Hist@3:{:.2%}, fil_Hist@1:{:.2%}'.
    #          format(mean_rank, mean_rank_reverse, Hist_10, Hist_3, Hist_1, fil_mean_rank, fil_mean_rank_reverse, fil_Hist_10, fil_Hist_3, fil_Hist_1))





if __name__=='__main__':
    get_triples_from_all_datasets('data/FB15k')