import codecs
import sys
import pickle

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
        names = ['train', 'test', 'valid']
        
        for name in names:
            fr = codecs.open(self.root + '/' + name + '2index.txt', 'r', encoding='utf-8')
            # print(name)
            for line in fr.readlines()[1:]:
                # print(line.strip())
                h, r, t = line.strip().split('\t')
                h, r, t = int(h), int(r), int(t)
                if (h, r) not in self.left:
                    self.left[(h, r)] = []
                if (r, t) not in self.right:
                    self.right[(r, t)] = []
                self.left[(h, r)].append(t)
                self.right[(r, t)].append(h)
            fr.close()

        for name in names:
            fr = codecs.open(self.root + '/' + name + '2index.txt', 'r', encoding='utf-8')
            fw_left = codecs.open(self.root + '/' + name + '.pkl', 'wb')
            # fw_left.write(fr.readline())
            # print(fr, self.root + '/' + name + '2index.txt')
            dataset_ = dict()
            for n, line in enumerate(fr.readlines()[1:]):
                # print(line.strip())
                h, r, t = tuple(line.strip().split())
                h, r, t = int(h), int(r), int(t)
                dataset_[n] = {'h':h, 'r':r, 't':t, 't_multi_1':self.left[(h, r)]}
                # fw_left.write(line.strip() + '\t' + str(self.left[(h, r)]) + '\n')
           
            pickle.dump(dataset_, fw_left)
            fr.close()
            fw_left.close()


             

            
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
        # create index of (h, r)-->
        
        
        
        
        fw11 = codecs.open(self.root + '/' + '1-1.pkl', 'wb')
        fw1n = codecs.open(self.root + '/' + '1-N.pkl', 'wb')
        fwn1 = codecs.open(self.root + '/' + 'N-1.pkl', 'wb')
        fwnn = codecs.open(self.root + '/' + 'N-N.pkl', 'wb')
        fr = codecs.open(self.root + '/' + 'test2index.txt', 'r', encoding='utf-8')
        s11 = 0
        sn1 = 0
        s1n = 0
        snn = 0
        tmp = fr.readlines()[1:]
        for line in tmp:
            h, r, t = line.strip().split('\t')
            h, r, t = int(h), int(r), int(t)
            left_n = float(self.right_rel[r]) / float(self.right_tot[r])
            right_n = float(self.left_rel[r]) / float(self.left_tot[r])
            # print(left_n, right_n)
            if right_n < 1.5 and left_n < 1.5:
                s11 += 1
            elif right_n < 1.5 and left_n >= 1.5:
                sn1 += 1
            elif right_n >= 1.5 and left_n < 1.5:
                s1n += 1
            elif right_n >= 1.5 and left_n >= 1.5:
                snn += 1
        # fw11.write(str(s11) + '\n')
        # fw1n.write(str(s1n) + '\n')
        # fwn1.write(str(sn1) + '\n')
        # fwnn.write(str(snn) + '\n')
        dataset_11 = dict()
        dataset_1n = dict()
        dataset_n1 = dict()
        dataset_nn = dict()
        print('1-1:', s11, '1-N:', s1n, 'N-1:', sn1, 'N-N:', snn)
        for line in tmp:
            h, r, t = line.strip().split('\t')
            h, r, t = int(h), int(r), int(t)
            # print(h, r, t)
            left_n = self.right_rel[r] / self.right_tot[r]
            right_n = self.left_rel[r] / self.left_tot[r]
            if right_n < 1.5 and left_n < 1.5:
                dataset_11[n] = {'h':h, 'r':r, 't':t, 't_multi_1':self.left[(h, r)]}
                # fw11.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\t' + str(self.left[(h, r)]) + '\n')
            elif right_n < 1.5 and left_n >= 1.5:
                dataset_n1[n] = {'h':h, 'r':r, 't':t, 't_multi_1':self.left[(h, r)]}
                # fwn1.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\t' + str(self.left[(h, r)]) + '\n')
            elif right_n >= 1.5 and left_n < 1.5:
                dataset_1n[n] = {'h':h, 'r':r, 't':t, 't_multi_1':self.left[(h, r)]}
                # fw1n.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\t' + str(self.left[(h, r)]) + '\n')
            elif right_n >= 1.5 and left_n >= 1.5:
                dataset_nn[n] = {'h':h, 'r':r, 't':t, 't_multi_1':self.left[(h, r)]}
                # fwnn.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\t' + str(self.left[(h, r)]) + '\n')
        pickle.dump(dataset_11, fw11)
        pickle.dump(dataset_1n, fw1n)
        pickle.dump(dataset_n1, fwn1)
        pickle.dump(dataset_nn, fwnn)
        
        
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
    # def create_h2t(self):
    #     files = ['train2index.txt', 'test2index.txt', 'valid2index.txt']
    #     for f in files:
    #         fr = codecs.open(self.root + '/' + f, 'r', encoding='utf-8')
    #         tmp = fr.readlines()
    #         for line in tmp[1:]:
    #             h, r, t = tuple(line.strip().split())
    #             print(h, r, t)

    
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
            fw.close()
            print(type, ', tot_num:', n + 1)
        print('create train2index, test2index, valid2index ok!')
        '''
            create NN
        '''
        self.create_NN()
        print('create NN ok!')


    


        


    





if __name__=='__main__':
    args1 = {'path':'countries_S1', 'name':'', 'name_list':['countriess1_train.txt', 'countriess1_test.txt', 'countriess1_valid.txt']}
    args2 = {'path':'toy', 'name':'', 'name_list':['toy_train.txt', 'toy_test.txt', 'toy_valid.txt']}
    args3 = {'path':'wordnet-mlj12', 'name':'', 'name_list':['wordnet-mlj12-train.txt', 'wordnet-mlj12-test.txt', 'wordnet-mlj12-valid.txt']}
    args4 = {'path':'FB15k-237', 'name':'', 'name_list':['fb237-train.txt', 'fb237-test.txt', 'fb237-valid.txt']}
    args5 = {'path':'FB15k', 'name':'', 'name_list':['freebase_mtr100_mte100-train.txt', 'freebase_mtr100_mte100-test.txt', 'freebase_mtr100_mte100-valid.txt']}
    argsn = [args1, args2, args3, args4, args5]
    for args in argsn[:]:
        t = pre_process_data(args['path'], args['name'], args['name_list'])
        t.run()

        fr = codecs.open(args['path'] + '/' + 'train' + '.pkl', 'rb')
        t = pickle.load(fr)
        print(len(t))
    
