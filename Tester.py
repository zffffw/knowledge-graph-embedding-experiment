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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_data_loader = test_data_loader
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.head_predict, self.tail_predict = get_triples_from_all_datasets(datasets_param.d[params.data]['root'])
        self.params = params
        self.test_batch_size = params.test_batch_size
        self.save_root = 'checkpoint/' + params.model + '/' + params.data + '/'
        self.save_best_name = self.save_root + self.params.model + '.emb_' +  str(self.params.embedding_dim)\
                 +'.lr_' + str(self.params.lr) + '.data_' + self.params.data + '.optim_' +  \
                     self.params.optimizer + '.loss_' + self.params.loss +'.batch_size_' +  \
                     str(self.params.batch_size) 
        if self.params.cluster_ent_name:
            self.save_best_name +=  '.cluster_ent_' + self.params.cluster_ent_name 
        if self.params.cluster_rel_name:
            self.save_best_name +=  '.cluster_rel_' + self.params.cluster_rel_name
        self.save_best_name += '.best.ckpt.Test.txt'
    
    def replace_all_entities(self, h, r, t, rtype = 'head'):
        if rtype == 'head':
            h = torch.arange(0, self.ent_tot).reshape(1, -1).repeat(r.shape[0], 1)
            r = r.reshape(-1, 1).repeat(1, self.ent_tot)
            t = t.reshape(-1, 1).repeat(1, self.ent_tot)
        elif rtype == 'tail':
            t = torch.arange(0, self.ent_tot).reshape(1, -1).repeat(r.shape[0], 1)
            r = r.reshape(-1, 1).repeat(1, self.ent_tot)
            h = h.reshape(-1, 1).repeat(1, self.ent_tot)
        return h.to(self.device), r.to(self.device), t.to(self.device)

    def test_run(self, hist=[1, 3, 10], mode=None):
        print("#####running test#####")
        self.model.eval()
        filter_mrr_1 = 0.0
        filter_mr_1 = 0.0
        Hist_filter_1 = [[] for i in range(10)]
        filter_mrr_2 = 0.0
        filter_mr_2 = 0.0
        Hist_filter_2 = [[] for i in range(10)]
        filter_mr = 0.0
        filter_mrr = 0.0
        Hist_filter = [[] for i in range(10)]
        tot = 0
        for n, data_val in enumerate(self.test_data_loader):
            #only use h, r, t, label
            h, r, t, tail_label, head_label = data_val['h'], data_val['rel'], data_val['t'], data_val['t_multi_1'], data_val['h_multi_1']
            h, r, t = h.to(self.device), r.to(self.device), t.to(self.device)
            cur_batch_size = h.shape[0]
            tot += int(cur_batch_size) # tot test size
            all_h_1, all_r_1, all_t_1 = self.replace_all_entities(h, r, t, 'tail')
            all_h_2, all_r_2, all_t_2 = self.replace_all_entities(h, r, t, 'head')
            for i in range(cur_batch_size):
                tmp1 = self.model.predict(all_h_1[i], all_r_1[i], all_t_1[i], isEval=True)
                tmp2 = self.model.predict(all_h_2[i], all_r_2[i], all_t_2[i], isEval=True)
                if i == 0:
                    all_score_filter_1 = tmp1
                    all_score_filter_2 = tmp2
                else:
                    all_score_filter_1 = torch.cat((all_score_filter_1, tmp1), -1)
                    all_score_filter_2 = torch.cat((all_score_filter_2, tmp2), -1)
        
            # print(all_score_raw)
            all_score_filter_1 = all_score_filter_1.reshape(cur_batch_size, -1).cpu()
            all_score_filter_2 = all_score_filter_2.reshape(cur_batch_size, -1).cpu()
            # print(all_score_filter.shape)
            # print(all_score_raw)
            label_1 = []
            label_2 = []

            # print(all_score_raw.shape, all_score_filter.shape)
            # print(target)
            for i in tail_label:
                label_1.append(eval(i))
            for i in head_label:
                label_2.append(eval(i))
            # print(len(label_1), len(label_2))

            for i in range(cur_batch_size):
                target1 = t[i].item()
                target2 = h[i].item()
                # target = t[i].item()
                tmp1 = all_score_filter_1[i][target1].item()
                tmp2 = all_score_filter_2[i][target2].item()
                if self.params.loss in ['margin']:
                    all_score_filter_1[i][label_1[i]] = 1000000
                    all_score_filter_2[i][label_2[i]] = 1000000
                elif self.params.loss in ['bce', 'ce', 'sfmargin', 'sploss']:
                    all_score_filter_1[i][label_1[i]] = -1000000
                    all_score_filter_2[i][label_2[i]] = -1000000
                all_score_filter_1[i][target1] = tmp1
                all_score_filter_2[i][target2] = tmp2
            descending = True if self.params.loss in ['bce', 'ce', 'sfmargin', 'sploss'] else False
            sorted_data_filter_1, indices_filter_1 = torch.sort(all_score_filter_1, -1, descending=descending)
            sorted_data_filter_2, indices_filter_2 = torch.sort(all_score_filter_2, -1, descending=descending)
            # print(sorted_data_filter.shape)
            for i in range(cur_batch_size):
                target1 = t[i].item()
                target2 = h[i].item()
                # target = t[i].item()
                filter_rank_1 = np.where(indices_filter_1[i] == target1)[0][0] + 1.0
                filter_rank_2 = np.where(indices_filter_2[i] == target2)[0][0] + 1.0
                # print(h[i], r[i], t[i], raw_rank, filter_rank, len(label_[i]))
                # input()
                filter_mr_1 += filter_rank_1
                filter_mrr_1 += 1.0 / (float(filter_rank_1))
                filter_mr_2 += filter_rank_2
                filter_mrr_2 += 1.0 / (float(filter_rank_2))
                filter_mr += filter_rank_1 + filter_rank_2
                filter_mrr += 1.0 / (float(filter_rank_1)) + 1.0 / (float(filter_rank_2))
                for k in range(10):
                    if filter_rank_1 <= float(k + 1):
                        Hist_filter[k].append(1.0)
                        Hist_filter_1[k].append(1.0)
                    else:
                        Hist_filter[k].append(0.0)
                        Hist_filter_1[k].append(0.0)
                    if filter_rank_2 <= float(k + 1):
                        Hist_filter[k].append(1.0)
                        Hist_filter_2[k].append(1.0)
                    else:
                        Hist_filter[k].append(0.0)
                        Hist_filter_2[k].append(0.0)
            # print("filter_mrr1:{:.3f},filter_mrr2:{:.3f},filter_mrr:{:.3f}".format(filter_mrr_1/tot, filter_mrr_2/tot, filter_mrr/tot/2), end='\r')
            print("{:<5}, filter_mr:{:<.1f} filter_mrr:{:.3f}, Hist10:{:.5f}  {}----{}, {}, {}\                    \
                    ".format(tot,  filter_mr / tot /2, filter_mrr / tot/2, np.mean(Hist_filter[9]), self.params.model, self.params.data, self.params.cluster_ent_name, self.params.cluster_rel_name), end='\r')
        tot *= 2
        print('\n###{}###'.format(mode))
        print("# filter MR      :{:.6f}".format(filter_mr / tot))
        print("# filter MR tail :{:.6f}".format(filter_mr_1 / tot*2))
        print("# filter MR head :{:.6f}".format(filter_mr_2 / tot*2))
        print("# filter MRR     :{:.6f}".format(filter_mrr / tot))
        print("# filter MRR tail:{:.6f}".format(filter_mrr_1 / tot*2))
        print("# filter MRR head:{:.6f}".format(filter_mrr_2 / tot*2))
        
        
        if mode == 'test':
            self.fw_log = open(self.save_best_name, 'a+', encoding='utf-8')
            self.fw_log.write('\n###{}###\n'.format(mode))
            self.fw_log.write("# filter MR:{:.6f}\n".format(filter_mr / tot))
            self.fw_log.write("# filter MRR:{:.6f}\n".format(filter_mrr / tot))
        for i in hist:
            print("# filter Hist@{}      : {:.6f}".format(i, np.mean(Hist_filter[i - 1])))
            print("# filter Hist tail@{} : {:.6f}".format(i, np.mean(Hist_filter_1[i - 1])))
            print("# filter Hist head@{} : {:.6f}".format(i, np.mean(Hist_filter_2[i - 1])))
            if mode == 'test':
                self.fw_log.write("# filter Hist@{} : {:.3f}\n".format(i, np.mean(Hist_filter[i - 1])))
        if mode == 'test':
            self.fw_log.write("{}\t{}\t{}\t{}\t{}\n".format(filter_mr / tot, filter_mrr / tot, np.mean(Hist_filter[9]),\
            np.mean(Hist_filter[2]), np.mean(Hist_filter[0])))
            self.fw_log.write("times:{} b1:{} b2:{}\n".format(self.params.times, self.params.b1, self.params.b2))
        return filter_mrr / tot, Hist_filter
            

                

    




if __name__=='__main__':
    get_triples_from_all_datasets('data/FB15k')