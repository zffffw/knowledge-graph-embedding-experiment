
import argparse
from Trainer import Trainer
from Tester import Tester
from model import *
from utils import *
from data.dataLoader import *
from config import *



parser = argparse.ArgumentParser(description='model')

parser.add_argument("--data", type=str, default="FB15k",
                    help="which dataset to use? FB15k/...")
parser.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing value to use. Default: 0.1")
parser.add_argument("--model", type=str, default="TransE",
                    help="Model to use")
parser.add_argument("--times", type=int, default=100,
                    help="Learning times")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size")
parser.add_argument("--check_step", type=int, default=50,
                    help="Interval of epochs to valid a checkpoint of the model?")
parser.add_argument("--save_step", type=int, default=10000,
                    help="Interval of epochs to save a checkpoint of the model?")
parser.add_argument("--eval_step", type=int, default=10,
                    help="Interval of epochs to evaluate the model?")
parser.add_argument('--eval_mode', type=str, default="head",
                    help='Evaluate on head and/or tail prediction?')
parser.add_argument("--negative_sample_size", type=int, default=1,
                    help="Number of negative samples to compare to for MRR/MR/Hit@10")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
parser.add_argument("--margin", type=float, default=1,
                    help="The margin between positive and negative samples in the max-margin loss")
parser.add_argument("--p_norm", type=int, default=1,
                    help="The norm to use for the distance metric")
parser.add_argument("--optimizer", type=str, default="SGD",
                    help="Which optimizer to use? SGD/Adam")
parser.add_argument("--embedding_dim", type=int, default=200,
                    help="Entity and relations embedding size")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Learning rate of the optimizer")
parser.add_argument("--momentum", type=float, default=0,
                    help="Momentum of the SGD optimizer")
parser.add_argument("--lmbda", type=float, default=0,
                    help="Regularization constant")
parser.add_argument("--debug", action='store_true',
                    help="Run the code in debug mode?")
parser.add_argument('--cuda', action='store_true', 
                    help='use cuda?')
parser.add_argument('--filter', action='store_true',
                    help='Filter the samples while evaluation')

parser.add_argument('--loss', type=str, default='margin',
                    help='which loss function?')
parser.add_argument('--test_flag', action='store_true',
                    help='test ?')
parser.add_argument('--train_flag', action='store_true',
                    help='train ?')
parser.add_argument('--sigmoid_flag', action='store_true',
                    help='use sigmoid ?')

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
params = parser.parse_args()

if params.debug:
    g = graph(datasets_param.d[params.data]['root'], 'test')
    g.create_graph()
    print(len(g.get_tail_neighbour(5957,27)))
    print(len(g.get_head_neighbour(8963, 37)))
    # for k in range(10000):
    #     a = g.get_neighbour(k, 0)
    #     tt = 0
    #     for i in a:
    #         h, r, t = i
    #         b = g.get_neighbour(t, 0)
    #         tt += len(b)
    #     print(k, tt)

else:

    device = 'cuda' if params.cuda else 'cpu'
    if params.loss == 'margin' and params.negative_sample_size == 0:
        raise Exception('the loss function is margin loss, please choose a negative sample size which is > 0')
    elif params.loss == 'bce' and params.negative_sample_size > 0:
        params.negative_sample_size = 0
        if not params.sigmoid_flag:
            raise Exception('must use sigmoid in model.')
    print(params)



    model = get_model(params.model, params.data, params.embedding_dim, params.p_norm, params.sigmoid_flag, params).to(device)
    loss = get_loss(params.loss, params.margin)
    if params.optimizer == 'SGD':
        opt = optim.SGD(model.parameters(), params.lr, params.momentum)
    elif params.optimizer == 'Adam':
        opt = optim.Adam(model.parameters(), params.lr)
    elif params.optimizer == 'AdaGrad':
        opt = optim.Adagrad(model.parameters(), params.lr)



    '''
        train
    '''


    train_data_loader = get_data_loader(params.data, params.batch_size, 'train', sample_size=params.negative_sample_size, param=params)
    valid_data_loader = get_data_loader(params.data, params.batch_size, 'valid', sample_size=params.negative_sample_size, param=params)
    ent_tot, rel_tot = dataset_param(params.data)
    trainer = Trainer(params, ent_tot, rel_tot, params.model, params.loss, train_data_loader, valid_data_loader, model, \
                loss, opt, params.batch_size, params.negative_sample_size, use_GPU=params.cuda, \
                     check_step=params.check_step, times=params.times, save_step=params.save_step) 
    if params.train_flag:
        print('[begin training]')
        trainer.run()

    '''
        test
    '''
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
                test_data_loader = get_data_loader(params.data, 100, tt, sample_size=0)
            except:
                print('no test data {}...'.format(tt))
                break
            ent_tot, rel_tot = dataset_param(params.data)
            tester = Tester(params, ent_tot, rel_tot, model, test_data_loader)
            print('run {} head.....'.format(tt))
            tester.test_run(type='head')