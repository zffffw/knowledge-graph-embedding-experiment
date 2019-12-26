import torch
import torch.nn as nn
import torch.nn.functional as F
import codecs

class Tester(object):
    def __init__(self, ent_tot, rel_tot, model, test_data_loader):
        self.device = 'cpu'
        self.model = model.to(self.device)
        self.test_data_loader = test_data_loader
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
    
    def test_one_step(self, test_data):
        res = self.model.predict(test_data[:, 0], test_data[:, 1], test_data[:, 2]).reshape(-1, 1)
        # print(res.shape, test_data.shape)
        return torch.cat((test_data.float(), res), 1)
    
    def get_rank(self, res, target, type='head'):
        for n, i in enumerate(res):
            if type == 'head':
                if int(i[0]) == target:
                    return n + 1
            elif type == 'tail':
                if int(i[2]) == target:
                    return n + 1

    def test_run(self, type='head'):
        self.model.eval()
        tot_rank = 0
        tot_rank_reverse = 0
        Hist_10 = 0
        Hist_3 = 0
        Hist_1 = 0
        for n, data_val in enumerate(self.test_data_loader):
            h, r, t = data_val['en1'], data_val['rel'], data_val['en2']
            if type == 'head':
                test_data = torch.cat((torch.arange(0, self.ent_tot).reshape(-1, 1), \
                    torch.Tensor([r]*self.ent_tot).long().reshape(-1, 1),\
                 torch.Tensor([t]*self.ent_tot).long().reshape(-1, 1)), 1)
                tmp = self.test_one_step(test_data.long().to(self.device)).detach().numpy()
                tmp = sorted(tmp, key=lambda x: x[3])
                rank = self.get_rank(tmp, h, type)

                
            elif type == 'tail':
                test_data = torch.cat((torch.Tensor([h]*self.ent_tot).long().reshape(-1, 1) ,\
                     torch.Tensor([r]*self.ent_tot).long().reshape(-1, 1),\
                 torch.arange(0, self.ent_tot).reshape(-1, 1)), 1)
                tmp = self.test_one_step(test_data.long().to(self.device)).detach().numpy()
                tmp = sorted(tmp, key=lambda x: x[3])
                rank = self.get_rank(tmp, t, type)
            print('mr:{:.3f}, mrr:{:.3f}, Hist@10:{:.2%}, Hist@3:{:.2%}, Hist@1:{:.2%}, {}/{}, {:.2%}'. \
                format(tot_rank / (n + 1), tot_rank_reverse / (n + 1), Hist_10 / (n + 1), Hist_3 / (n + 1),\
                     Hist_1 / (n + 1), n, len(self.test_data_loader), n/len(self.test_data_loader)), end='\r')
            tot_rank += rank
            tot_rank_reverse += 1.0/rank
            if rank <= 10:
                Hist_10 += 1
            if rank <= 3:
                Hist_3 += 1
            if rank <= 1:
                Hist_1 += 1
            
        n += 1
        mean_rank = tot_rank / n
        mean_rank_reverse = tot_rank_reverse / n
        Hist_10 /= n
        Hist_3 /= n
        Hist_1 /= n
        print('\n', type, 'mr:{:.3f}, mrr:{:.3f}, Hist@10:{:.2%}, Hist@3:{:.2%}, Hist@1:{:.2%}'.\
            format(mean_rank, mean_rank_reverse, Hist_10, Hist_3, Hist_1))





