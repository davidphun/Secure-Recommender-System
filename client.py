from numpy.core.fromnumeric import size
import numpy as np
from criterions import BPR
import torch
from ldp import LocalDP
import copy

class Client:
    def __init__(self, u_id, data_loader, model, args):
        self.u_id = u_id
        self.data_loader = data_loader
        # Populate the train/test data and unseen/seen items for the client
        self.train_data = data_loader.get_train_data_by_uid(u_id)
        self.test_data = data_loader.get_test_data_by_uid(u_id)
        self.seen_items = data_loader.get_train_items_by_uid(u_id)
        self.unseen_items = data_loader.get_unseen_items_by_uid(u_id)
        # Initialize the structure of model for the client, 
        # i.e., how many layers a model has, as well as the shape for each layer
        # Here, for the sake of simplicity, I initialize the model with the size equal to the global model, 
        # which is not practical for low-budget device due to its limited memory and computation power
        # In practice, each client's model will only contain non-embedding part 
        # and some item-embedding vectors that the client has interacted with as well as his/her user embedding
        self.model = copy.deepcopy(model)
        self.args = args
        self.item_generator = LocalDP(self.seen_items)
        # Keep track the indices of seen items during the submission phase to form the union set,
        # which will be used in the randomization process to decide whether the user should use which of those items as the rated items (positive items) to perform update
        self.pre_committed_items = None
        # Since I use FedAvg algorithm to train the global model; therefore, I need to keep track of 
        # the number of times a chosen item is involved in the local training process, i.e., how many training records are associated with the item
        # Last but not least, the # of training examples also needs to be recorded
        self.train_item_freq = {}
        self.n_train_records = 0
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.momentum, args.beta2), eps=args.eps, weight_decay=args.l2)
        # Create a variable to store the latest change in user embedding after updating for further evaluating the stop condition of the global model
        self.latest_u_embed_sum_square_change = torch.tensor(0)

    def commit_train_items(self):
        seen_item_set = np.random.choice(self.seen_items, replace=True, size=self.args.n_pos_per_user)
        unseen_item_set = np.random.choice(self.unseen_items, replace=True, size=self.args.n_neg_per_commit)
        self.pre_committed_items = np.concatenate([seen_item_set, unseen_item_set], axis=0)
        return self.pre_committed_items

    
    def compose_train_data_via_randomized_items(self, union_set):
        # Discriminate seen and unseen items in the union set for composing train data
        _, unseen_items = self.partition_seen_and_unseen_items(union_set)
        # Get random item indices via randomization process in which 
        # the pre-committed seen items will be paired with a certain number of unseen items in the union set to form the train data
        if not self.item_generator.is_optimized and \
            self.item_generator.need_optimize(len(unseen_items), self.args.n_neg_per_pos):
            self.item_generator.optimize_item_selection_rates(len(unseen_items), self.args.n_neg_per_pos)

        randomized_ids = self.item_generator.select_items_via_random_process(self.pre_committed_items)

        composed_data, selected_items = self.data_loader.sample_data_via_uid_and_lr_iid(self.u_id, randomized_ids, unseen_items, self.args.n_neg_per_pos, True)
        
        # Initialize the frequencies of items used in the training process
        self.initialize_train_item_freq(selected_items)
        
        return composed_data, selected_items
    
    def initialize_train_item_freq(self, items):
        for i_id in items:
            self.train_item_freq[i_id] = 0
        return

    def update_train_item_freq(self, items_torch):
        for i_id in items_torch:
            self.train_item_freq[i_id.item()] += 1


    def save_global_parameters(self, non_embeddings, i_embeddings):
        '''
            Update the local parameters by the requested global parameters
        '''
        # Load global item embeddings
        for i_id in i_embeddings.keys():
            # print(i_id)
            self.model.i_embeddings.weight.data[i_id] = i_embeddings[i_id].to(self.args.device)
            # print(self.model.i_embeddings.weight.data[i_id])
        # Load global non embeddings
        for l in range(self.model.n_hidden_layers):
            self.model.mlp_layers[l].weight.data = non_embeddings[l].weight.data.to(self.args.device)
            self.model.mlp_layers[l].bias.data = non_embeddings[l].bias.data.to(self.args.device)
        self.model.output_layer.weight.data = non_embeddings['output'].weight.data.to(self.args.device)
        self.model.output_layer.bias.data = non_embeddings['output'].bias.data.to(self.args.device)
        return

    def train_recommendation(self, data):
        X_train, y_train = data[:, :-1], data[:, -1].type(torch.float)
        # Create a variable to store the user embedding prior to each training process 
        # in order to compute the change after updating for further evaluating the stop condition of the global model
        self.prev_u_embedding_weight = self.model.u_embeddings.weight.data[self.u_id].cpu()
        # Update the frequency of each item used and count # of records in the training process
        self.update_train_item_freq(X_train[:, 1])
        self.update_train_item_freq(X_train[:, 2])
        self.n_train_records = len(y_train)
        # Initialize optimizer, criterion, and hyperparameters for training
        criterion = BPR()
        self.model.train()
        score_l_items = self.model(X_train[:, 0], X_train[:, 1])
        score_r_items = self.model(X_train[:, 0], X_train[:, 2])
        loss = criterion(score_l_items, score_r_items, y_train)
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    def get_parameter_update_data(self, global_non_embeddings, global_item_embeddings):
        item_set = list(self.train_item_freq.keys())
        non_embedding_update_data = {l:{'weight': 0, 'bias': 0} for l in range(self.model.n_hidden_layers)}
        non_embedding_update_data['output'] = {'weight': 0, 'bias': 0}
        i_embedding_update_data = {i_id: 0 for i_id in item_set}
        # If no training is performed, return 0 vectors as the data for updating the global non embedding and requested item embeddings...
        # which should not affect the global model performance as no update will be proceeded at the server side with respect to the update data sent by this user
        if self.n_train_records == 0:
            return non_embedding_update_data, i_embedding_update_data, self.latest_u_embed_sum_square_change 
        if self.args.fd_update_data_type == 'grad':
            #### If the we choose to submit gradients to the server, submit the weighted gradients
            # Compute the grad of non embeddings
            for l in self.model.get_mlp_layer_names():
                non_embedding_update_data[l]['weight'] = self.model.get_mlp_param_data(l, 'weight', True).cpu()
                non_embedding_update_data[l]['bias'] = self.model.get_mlp_param_data(l, 'bias', True).cpu()
                # non_embedding_update_data[l]['weight'] /= self.args.n_users_per_iter
                # non_embedding_update_data[l]['bias'] /= self.args.n_users_per_iter
                if self.args.loss_type == 'avg':
                    non_embedding_update_data[l]['weight'] *= self.n_train_records
                    non_embedding_update_data[l]['bias'] *= self.n_train_records
            # Compute the grad of item embeddings
            for i_id in item_set:
                i_embedding_update_data[i_id] =  self.model.i_embeddings.weight.grad.data[i_id].cpu()
                # i_embedding_update_data[i_id] /= self.args.n_users_per_iter
                if self.args.loss_type == 'avg':
                    i_embedding_update_data[i_id] *= self.train_item_freq[i_id]
        elif self.args.fd_update_data_type == 'change':
            #### If one chooses to submit the change of parameters after updating to the server, submit the weighted change
            # Compute the update data for non embeddings
            for l in self.model.get_mlp_layer_names():
                non_embedding_update_data[l]['weight'] = self.model.get_mlp_param_data(l, 'weight', False).cpu() - global_non_embeddings[l].weight.data
                non_embedding_update_data[l]['bias'] = self.model.get_mlp_param_data(l, 'bias', False).cpu() - global_non_embeddings[l].bias.data
                non_embedding_update_data[l]['weight'] /= self.args.n_users_per_iter
                non_embedding_update_data[l]['bias'] /= self.args.n_users_per_iter
                if self.args.loss_type == 'avg':
                    non_embedding_update_data[l]['weight'] *= self.n_train_records
                    non_embedding_update_data[l]['bias'] *= self.n_train_records
            # Compute the change in item embeddings
            for i_id in item_set:
                i_embedding_update_data[i_id] =  self.model.i_embeddings.weight.data[i_id].cpu() - global_item_embeddings[i_id]
                i_embedding_update_data[i_id] /= self.args.n_users_per_iter
                if self.args.loss_type == 'avg':
                    i_embedding_update_data[i_id] *= self.train_item_freq[i_id]
        # This is used to keep track of the sum of square change b4 and after updating user embedding
        # for further evaluation purpose
        u_embedding_change = self.model.u_embeddings.weight.data[self.u_id].cpu() - self.prev_u_embedding_weight
        self.latest_u_embed_sum_square_change = torch.sum(u_embedding_change * u_embedding_change)
        # Clear out the memory assigned for parameter gradients...
        self.optimizer.zero_grad(set_to_none=True)
        return non_embedding_update_data, i_embedding_update_data, self.latest_u_embed_sum_square_change

    def get_item_freq_and_n_records(self):
        '''
            Return the number of records used for the training process and 
            the freq of each selected item involves in the training records
        '''
        item_freq = self.train_item_freq
        n_records = self.n_train_records
        # Reset the memory for the new training process
        self.train_item_freq = {}
        self.n_train_records = 0
        return item_freq, n_records

    def commit_test_items(self, n_neg_items):
        test_items = self.test_data[:, 1].numpy()
        selected_unseen_items = np.random.choice(self.unseen_items, replace=False, size=n_neg_items)
        return np.concatenate([test_items, selected_unseen_items], axis=0)
    
    def compose_test_data_for_auc(self, union_set, n_right_per_left):
        test_items = self.test_data[:, 1].numpy()
        # Union set may contain both seen and unseen items
        # Therefore, we need to first partition the union set into set of known and unknown items
        # and then randomly select unknown items to form the test data with the test items
        _, right_items = self.partition_seen_and_unseen_items(union_set[union_set != test_items])
        data, _ = self.data_loader.sample_data_via_uid_and_lr_iid(self.u_id, test_items, right_items, n_right_per_left, False)
        return data[:, :-1]

    def compose_test_data_for_hit_and_ndcg(self, union_set, n_neg_items):
        test_items = self.test_data[:, 1].numpy()
        _, item_pool = self.partition_seen_and_unseen_items(union_set[union_set != test_items])
        data = self.data_loader.sample_single_item_per_record(self.u_id, test_items, item_pool, n_neg_items)
        return data, test_items

    def partition_seen_and_unseen_items(self, i_ids):
        seen_items, unseen_items = [], []
        for i_id in i_ids:
            if i_id in self.seen_items:
                seen_items.append(i_id)
            else:
                unseen_items.append(i_id)
        return np.array(seen_items), np.array(unseen_items)
