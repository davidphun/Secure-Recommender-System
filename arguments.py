import torch
class Arguments:
    def __init__(self):
        self.device = self.initialize_device()
        self.eps = 1e-8
        self.n_users_per_iter = 16 # Number of selected users per iteration
        self.n_pos_per_user = 8 # Number of positive items per iteration
        self.n_neg_per_pos = 8 # Number of negative items per positive item
        # Number of negative items submitted by a user to form the union set
        self.n_neg_per_commit = 8
        self.log_interval = 200
        #1e-3 for centralized setting
        #1e-2 for FL setting
        self.lr = 1e-2
        self.beta2 = 0.999 # Hyperparams for Adam optimizer
        self.momentum = 0.9
        #1e-3 for centralized setting
        #5e-3 for FL setting
        self.l2 = 5e-3 
        self.max_iter = 3000
        self.fd_update_data_type = 'grad' # change or grad
        self.fd_update_method = 'FedAdam' # FedAdam or FedAvg
        self.loss_type = 'sum' # sum or avg

    def initialize_device(self):
        '''
        Creates appropriate torch device for client operation.
        '''
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")