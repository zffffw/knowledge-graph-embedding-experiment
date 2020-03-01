
import argparse
from utils import *
from data.dataLoader import *
from config import *
from Trainer import Trainer
from Tester import Tester



# parser define
parser = argparse.ArgumentParser(description='model')
parser.add_argument("--data", type=str, default="FB15k",
                    help="which dataset to use? FB15k/wn/FB15k-237/toy/countries_S1")
parser.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing value to use. Default: 0.1")
parser.add_argument("--model", type=str, default="TransE",
                    help="which model to use TransE/DistMult/ConvE/ComplEx(Default:TransE)")
parser.add_argument("--times", type=int, default=100,
                    help="Learning epochs(Default:100)")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size")
parser.add_argument("--check_step", type=int, default=50,
                    help="Interval of epochs to valid a checkpoint of the model?")
parser.add_argument("--save_step", type=int, default=10000,
                    help="Interval of epochs to save a checkpoint of the model?")
parser.add_argument("--eval_step", type=int, default=50,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument('--eval_mode', type=str, default="head",
                    help='Evaluate on head and/or tail prediction?')
parser.add_argument("--negative_sample_size", type=int, default=1,
                    help="Number of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--margin", type=float, default=1.0,
                    help="The margin between positive and negative samples in the max-margin loss(Defalut: 1.0)")
parser.add_argument("--p_norm", type=int, default=1,
                    help="The norm to use for the distance metric(Default:1)")
parser.add_argument("--optimizer", type=str, default="Adam",
                    help="Which optimizer to use? SGD/Adam/Adagrad(Default:Adam)")
parser.add_argument("--embedding_dim", type=int, default=200,
                    help="Entity and relations embedding size")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate of the optimizer")
parser.add_argument("--momentum", type=float, default=0,
                    help="Momentum of the SGD optimizer")
parser.add_argument("--lmbda", type=float, default=0,
                    help="Regularization constant")
parser.add_argument('--cuda', type=float, default=0,
                    help="which gpu id to use.(Default:0, if -1 then use cpu)")
parser.add_argument('--loss', type=str, default='margin',
                    help='which loss function? margin/bce/ce. (Default: margin)')
parser.add_argument('--test_flag', action='store_true',
                    help='whether test or not')
parser.add_argument('--train_flag', action='store_true',
                    help='whrther train or not')
parser.add_argument('--sigmoid_flag', action='store_true',
                    help='whether to use sigmoid at the end or the model.')
parser.add_argument('--input_drop', type=float, default=0.2,
                    help='input dropout layer param for ConvE')
parser.add_argument('--hidden_drop', type=float, default=0.3,
                    help='hidden dropout layer param for ConvE')
parser.add_argument('--feat_drop', type=float, default=0.2,
                    help='feature map dropout layer param for ConvE')
parser.add_argument('--embedding-shape1', type=int, default=20,
                    help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
parser.add_argument('--hidden_size', type=int, default=9728,
                    help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
parser.add_argument('--use-bias', action='store_true', 
                    help='Use a bias in the convolutional layer. Default: True')
parser.add_argument('--mode', type=str, default='neg_sample',
                    help='train mode(neg_sample/1vsall/kvsall.(Default:neg_sample')
parser.add_argument('--regularize', type=float, default=0.0,
                    help='regularization parameter(Default:0.0)')
params = parser.parse_args()


device = torch.device("cuda:" + str(params.cuda) if params.cuda > -1 else "cpu")

# check training mode
if params.mode in ['kvsall', '1vsall']:
    if params.negative_sample_size > 0:
        params.negative_sample_size = 0
        print('warning: if you use kvsall/1vsall, the negative sample size is not necessary.')
elif params.mode in ['neg_sample']:
    if params.negative_sample_size == 0:
        raise Exception('the training mode is neg_sample, please choose a negative sample size which is > 0')
# check loss function
if params.loss == 'margin':
    if params.mode != 'neg_sample':
        raise Exception('margin loss function must use [neg_sample] training mode.')
elif params.loss == 'bce':
    if not params.sigmoid_flag:
        raise Exception('must use sigmoid in model.')
elif params.loss == 'ce':
    if params.mode != '1vsall':
        raise Exception('croeeentropyloss function must use [1vsall] training mode.')



    
print(params)

train_data_loader = get_data_loader(params, 'train')
valid_data_loader = get_data_loader(params, 'valid')


model = get_model(params)
opt = get_optimizer(model, params)

print(model)


# '''
#     train
# '''


ent_tot, rel_tot = dataset_param(params.data)
trainer = Trainer(params, ent_tot, rel_tot, train_data_loader, valid_data_loader, model, opt) 
if params.train_flag:
    print('[begin training]')
    trainer.run()

# '''
#     test
# '''
if params.test_flag:
    print('[begin testing]')
    
    model_load_name = trainer.save_best_name
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_load_name))
    else:
        model.load_state_dict(torch.load(model_load_name, map_location=torch.device('cpu')))
    ttype = ['test', '1-1', '1-N', 'N-1', 'N-N']
    for tt in ttype:
        try:
            test_data_loader = get_data_loader(params, tt)
            ent_tot, rel_tot = dataset_param(params.data)
            tester = Tester(params, ent_tot, rel_tot, model, test_data_loader)
            tester.test_run(type='head')
        except:
            print('no test data {}...'.format(tt))
        
        