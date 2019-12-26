import codecs
import sys

class pre_process_data:
    def __init__(self,root, load_name='', name_list = []):
        self.root = root
        self.name_list = name_list
        self.load_name = load_name
        self.item2index = {'entity':{}, 'relation':{}}
        self.index2item = {'entity':{}, 'relation':{}}
        self.left = {}
        self.right = {}
        self.left_rel = {}
        self.right_rel = {}
        self.left_tot = {}
        self.right_tot = {}
        
    def get_dictionary(self, choice=[0, 2]):
        cnt_e = 0
        cnt_r = 0
        for name in self.name_list:
            # print(self.root + '/' + name)
            f = codecs.open(self.root + '/' + name, 'r', encoding='utf-8')
            for line in f.readlines():
                tmp = line.strip().split('\t')
                for num in choice:
                    if num in [0, 2]:
                        if tmp[num] not in self.item2index['entity']:
                            self.item2index['entity'][tmp[num]] = cnt_e 
                            self.index2item['entity'][cnt_e] = tmp[num]
                            cnt_e += 1
                    else:
                        if tmp[num] not in self.item2index['relation']:
                            self.item2index['relation'][tmp[num]] = cnt_r
                            self.index2item['relation'][cnt_r] = tmp[num]
                            cnt_r += 1
        print('entity:', len(self.item2index['entity']), ', relation', len(self.item2index['relation']))


    def create_NN(self):
        names = ['train2index.txt', 'test2index.txt', 'valid2index.txt']
        for name in names:
            fr = codecs.open(self.root + '/' + name, 'r', encoding='utf-8')
            for line in fr.readlines()[1:]:
                h, r, t = line.strip().split('\t')
                if (h, r) not in self.left:
                    self.left[(h, r)] = []
                if (r, t) not in self.right:
                    self.right[(r, t)] = []
                self.left[(h,r)].append(t)
                self.right[(r, t)].append(h)
            
        for (h, r) in self.left:
            if r not in self.left_rel:
                self.left_rel[r] = 0.0
                self.left_tot[r] = 0.0
            self.left_rel[r] += len(self.left[(h, r)])
            self.left_tot[r] += 1.0
        for (r, t) in self.right:
            if r not in self.right_rel:
                self.right_rel[r] = 0.0
                self.right_tot[r] = 0.0
            self.right_rel[r] += len(self.right[(r, t)])
            self.right_tot[r] += 1.0
        # print(len(self.left), len(self.right))
        
        
        fw11 = codecs.open(self.root + '/' + '1-1.txt', 'w', encoding='utf-8')
        fw1n = codecs.open(self.root + '/' + '1-N.txt', 'w', encoding='utf-8')
        fwn1 = codecs.open(self.root + '/' + 'N-1.txt', 'w', encoding='utf-8')
        fwnn = codecs.open(self.root + '/' + 'N-N.txt', 'w', encoding='utf-8')
        fr = codecs.open(self.root + '/' + 'test2index.txt', 'r', encoding='utf-8')
        s11 = 0
        sn1 = 0
        s1n = 0
        snn = 0
        tmp = fr.readlines()[1:]
        for line in tmp:
            h, r, t = line.strip().split('\t')
            left_n = self.right_rel[r] / self.right_tot[r]
            right_n = self.left_rel[r] / self.left_tot[r]
            if right_n < 1.5 and left_n < 1.5:
                s11 += 1
            elif right_n < 1.5 and left_n >= 1.5:
                sn1 += 1
            elif right_n >= 1.5 and left_n < 1.5:
                s1n += 1
            elif right_n >= 1.5 and left_n >= 1.5:
                snn += 1
        fw11.write(str(s11) + '\n')
        fw1n.write(str(s1n) + '\n')
        fwn1.write(str(sn1) + '\n')
        fwnn.write(str(snn) + '\n')
        print('1-1:', s11, '1-N:', s1n, 'N-1:', sn1, 'N-N:', snn)
        for line in tmp:
            h, r, t = line.strip().split('\t')
            left_n = self.right_rel[r] / self.right_tot[r]
            right_n = self.left_rel[r] / self.left_tot[r]
            if right_n < 1.5 and left_n < 1.5:
                fw11.write(h + '\t' + r + '\t' + t + '\n')
            elif right_n < 1.5 and left_n >= 1.5:
                fwn1.write(h + '\t' + r + '\t' + t + '\n')
            elif right_n >= 1.5 and left_n < 1.5:
                fw1n.write(h + '\t' + r + '\t' + t + '\n')
            elif right_n >= 1.5 and left_n >= 1.5:
                fwnn.write(h + '\t' + r + '\t' + t + '\n')

        
        # print(len(self.left_rel), len(self.right_rel))

        
        # print(self.right_rel)



        
    '''
    args: entity: save entity to index as ***_entity2index.txt.
          relation: save relation to index as ***_relation2index.txt.
    '''

    def create_item2index(self, type='entity'):
        if type not in ['entity', 'relation']:
            print('please set arg between \'entity\' or \'relation\'')
            return False
        fw_i2i = codecs.open(self.root + '/' + self.load_name + '_' + type + '2index.txt', 'w', encoding='utf-8')
        for n, item in enumerate(self.item2index[type]):
            fw_i2i.write(item + '\t' + str(self.item2index[type][item]))   
            if n != len(self.item2index[type]) - 1:
                fw_i2i.write('\n')


        
    def load_item2index(self, type='entity'):
        if type not in ['entity', 'relation']:
            print('please set arg between \'entity\' or \'relation\'')
            return False
        fr_i2i = codecs.open(self.root + '/' + self.load_name + '_' + type + '2index.txt', 'r', encoding='utf-8')
        self.item2index[type] = {}
        for line in fr_i2i.readlines():
            tmp = line.strip().split('\t')
            self.item2index[type][tmp[0]] = int(tmp[1])
        return self.item2index[type]
    
    def load_index2item(self, type='entity'):
        if type not in ['entity', 'relation']:
            print('please set arg between \'entity\' or \'relation\'')
            return False
        fr_i2i = codecs.open(self.root + '/' + self.load_name + '_' + type + '2index.txt', 'r', encoding='utf-8')
        self.index2item[type] = {}
        for line in fr_i2i.readlines():
            tmp = line.strip().split('\t')
            self.index2item[type][int(tmp[1])] = tmp[0]
        return self.index2item[type]
    
    def run(self):
        '''
            read raw data
        '''
        self.get_dictionary(choice=[0, 1, 2])
        print('read data:'+','.join(self.name_list)+' ok!')
        '''
            create entity2index and relation2index
        '''
        self.create_item2index('entity')
        self.create_item2index('relation')
        print('create item2index ok!')
        '''
            create train2index, test2index, valid2index
        '''
        data_type = ['train', 'test', 'valid']
        for type in data_type:
            name = [i for i in self.name_list if type in i][0]
            fr = codecs.open(self.root + '/' + name, 'r', encoding='utf-8')
            fw = codecs.open(self.root + '/' + type + '2index.txt', 'w', encoding='utf-8')
            tmp = fr.readlines()
            fw.write(str(len(tmp)) + '\n')
            for n, line in enumerate(tmp):
                tmp = line.strip().split('\t')
                head = str(self.item2index['entity'][tmp[0]])
                relation = str(self.item2index['relation'][tmp[1]])
                tail = str(self.item2index['entity'][tmp[2]])
                fw.write(head + '\t' + relation + '\t' + tail + '\n')
            print(type, ', tot_num:', n + 1)
        print('create train2index, test2index, valid2index ok!')
        '''
            create NN
        '''
        self.create_NN()
        print('create NN ok!')


    


        


    





if __name__=='__main__':
    args = {'path':'countries_S1', 'name':'', 'name_list':['countriess1_train.txt', 'countriess1_test.txt', 'countriess1_valid.txt']}
    args = {'path':'toy', 'name':'', 'name_list':['toy_train.txt', 'toy_test.txt', 'toy_valid.txt']}
    args = {'path':'wordnet-mlj12', 'name':'', 'name_list':['wordnet-mlj12-train.txt', 'wordnet-mlj12-test.txt', 'wordnet-mlj12-valid.txt']}
    # args = {'path':'FB15k', 'name':'', 'name_list':['freebase_mtr100_mte100-train.txt', 'freebase_mtr100_mte100-test.txt', 'freebase_mtr100_mte100-valid.txt']}
    t = pre_process_data(args['path'], args['name'], args['name_list'])

    # t = pre_process_data('FB15k', '', ['freebase_mtr100_mte100-train.txt', 'freebase_mtr100_mte100-test.txt', 'freebase_mtr100_mte100-valid.txt'])
    t.run()

