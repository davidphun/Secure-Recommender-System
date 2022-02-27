from train import centralized_train
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from model import SVD_MF, NCF
from data_loader import DataLoader
import const
from arguments import Arguments
import numpy as np
from train import centralized_train, federated_train
import time
import matplotlib.pyplot as plt
from evaluation import evaluate_by_auc, evaluate_by_hit_and_ndcg
import pickle

torch.manual_seed(123)
np.random.seed(123) # For reproducibility

if __name__ == '__main__':
    args = Arguments()
    train_data = torch.load('train_data.pt')
    test_data = torch.load('test_data.pt')
    with open('unseen_items.pickle', 'rb') as f:
        unseen_data = pickle.load(f)

    start_t = time.time()
    data_loader = DataLoader(args, train_data, test_data, unseen_data)
    print('Time to initialize data loader: {}s'.format(time.time() - start_t))
    
    # #### SVD MF
    
    # # dims = [2, 8, 16, 32, 48, 64, 128]
    # avg_aucs = []
    # avg_hits = []
    # avg_ndcgs = []
    # dims = [128]
    
    # for d in dims:
    #     print('-'*10 + '{} dims'.format(d) + '-'*10)
    #     model = SVD_MF(d).to(args.device)
    #     optimized_model = centralized_train(model, data_loader, args)
    #     avg_aucs.append(
    #         evaluate_by_auc(optimized_model, data_loader, np.arange(const.N_USERS), 100)
    #         )
    #     avg_hit, avg_ndcg = evaluate_by_hit_and_ndcg(optimized_model, data_loader, np.arange(const.N_USERS), 10, 100)
    #     avg_hits.append(avg_hit)
    #     avg_ndcgs.append(avg_ndcg)

    # plt.plot(dims, avg_aucs, label='AUC')
    # plt.plot(dims, avg_hits, label='HR@{}'.format(10))
    # plt.plot(dims, avg_ndcgs, label='NDCG@{}'.format(10))

    # plt.xlabel('Dimensions')
    # plt.ylabel('0.0-1.0')
    # plt.title('Regularized SVD MF')
    # plt.legend()

    # plt.show()
    
    #### NCF Centralized Training
    #dims = [64, 128, 256, 512]
    # dims = [128]
    # avg_aucs = []
    # avg_hits = []
    # avg_ndcgs = []
    # for d in dims:
    #     hyperparams = {
    #         'n_latent_dims': d,
    #         'n_hidden_layers': 3,
    #         'n_users': const.N_USERS,
    #         'n_items': const.N_ITEMS
    #     }
    #     print('-'*10 + '{} dims'.format(d) + '-'*10)
    #     model = NCF(hyperparams).to(args.device)
    #     optimized_model = centralized_train(model, data_loader, args)
    #     avg_aucs.append(
    #         evaluate_by_auc(optimized_model, data_loader, np.arange(const.N_USERS), 100)
    #         )
    #     avg_hit, avg_ndcg = evaluate_by_hit_and_ndcg(optimized_model, data_loader, np.arange(const.N_USERS), 10, 100)
    #     avg_hits.append(avg_hit)
    #     avg_ndcgs.append(avg_ndcg)

    # plt.plot(dims, avg_aucs, label='AUC')
    # plt.plot(dims, avg_hits, label='HR@{}'.format(10))
    # plt.plot(dims, avg_ndcgs, label='NDCG@{}'.format(10))

    # plt.xlabel('Dimensions')
    # plt.ylabel('0.0-1.0')
    # plt.title('Regularized NCF')
    # plt.legend()
    # plt.show()

    #### NCF Federated Training
    hyperparams = {
            'n_latent_dims': 128,
            'n_hidden_layers': 3,
            'n_users': const.N_USERS,
            'n_items': const.N_ITEMS
    }
    model = NCF(hyperparams).to(args.device)
    federated_train(model, data_loader, args)