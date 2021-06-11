import argparse
import sys


def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('ASTGN')
    parser.add_argument('--dataset', type=str, default="mooc", choices=["wiki", "reddit","enron","socialevolve","uci","mooc"],
                        help='the name of dataset')
    parser.add_argument('--tasks', type=str, default="LP", choices=["LP", "NC"],
                        help='task name link prediction or node classification')

    #-------------------------------------------------------------------------------------
    parser.add_argument('--causual_sampling', type=bool,default=True, help='make sample path valid or not')
    parser.add_argument('--bandit',type=str, default="cDyBandit", choices=["NoBandit", "OriBandit","cDyBandit"],
                        help='the type of bandit')
    #parameter of original Bandit
    parser.add_argument('--eta', type=float, default="0.4",help='the parameter of bandit sampler')
    parser.add_argument('--T', type=int, default="40", help='the parameter of bandit sampler')
    #parameter of dynamic cBandit using LinUCB algorithm
    parser.add_argument('--alpha', type=float, default="0.5", help='exploration term')
    parser.add_argument('--delta', type=float, default="0.5", help='change threshold')

    # -------------------------------------------------------------------------------------
    # hyperparameter
    parser.add_argument('--fanouts', type=int, default=[],
                        help='the list of the numbers of sampling neighbors in each layer')
    parser.add_argument('--batch_size', type=int, default=768, help='Batch_size')
    parser.add_argument('--emb_dimension', type=int, default=100, help='emb_dimension')
    parser.add_argument('--time_dimension', type=int, default=100, help='dimension of time-encoding')
    parser.add_argument('--edge_feat_dim', type=int, default=172, help='Dimensions of the edge feature')
    parser.add_argument('--node_feat_dim', type=int, default=100, help='Dimensions of the node feature')
    parser.add_argument('--n_degrees', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of network layers')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')

    # -------------------------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=-1, help='Random Seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--warmup', action='store_true', help='')


    try:
        args = parser.parse_args()
        assert args.n_workers == 0, "n_worker must be 0, etherwise dataloader will cause bug and results very bad performance (this bug will be fixed soon)"
        args.feat_dim = 172
        args.no_time = True
        # args.no_pos = True

    except:
        parser.print_help()
        sys.exit(0)

    return args

