

class datasets_param:
    d = {'FB15k':{
        'root': 'data/FB15k',
        'name_list': ['freebase_mtr100_mte100-test.txt', 'freebase_mtr100_mte100-train.txt', 'freebase_mtr100_mte100-valid.txt'],
        'ent_tot': 14951,
        'rel_tot': 1345
    }, 'wn':{

    }, 'countries_S1':{
        'root': 'data/countries_S1',
        'name_list': ['countriess1_train.txt', 'countriess1_test.txt', 'countriess1_valid.txt'],
        'ent_tot': 271,
        'rel_tot': 2

    }, 'toy':{
        'root': 'data/toy',
        'name_list': ['toy_train.txt', 'toy_test.txt', 'toy_valid.txt'],
        'ent_tot': 16,
        'rel_tot': 9
    }}

class fb15k:
    '''
        the path is based on data/
    '''
    root = '/Users/zhang/Documents/newKGE/data/FB15k'
    name_list = ['freebase_mtr100_mte100-test.txt', 'freebase_mtr100_mte100-train.txt', 'freebase_mtr100_mte100-valid.txt']
    load_name = 'fb15k'
    ent_tot = 14951
    rel_tot = 1345

class wn:
    '''
        the path is based on data/
    '''
    root = '/Users/zhang/Documents/newKGE/data/data/wordnet-mlj12'
    name_list = ['wordnet-mlj12-test.txt', 'wordnet-mlj12-train.txt', 'wordnet-mlj12-valid.txt']
    load_name = 'wnmlj12'
    ent_tot = 100
    rel_tot = 100


class countries_s1:
    root = '/Users/zhang/Documents/newKGE/data/countries_S1'
    name_list = ['countriess1_train.txt', 'countriess1_test.txt', 'countriess1_valid.txt']
    ent_tot = 271
    rel_tot = 2


class toy:
    root = '/Users/zhang/Documents/newKGE/data/toy'
    name_list = ['toy_train.txt', 'toy_test.txt', 'toy_valid.txt']
    ent_tot = 16
    rel_tot = 9