import torch
import torch.nn.functional as F
import numpy as np
import const
from torch.autograd import Variable

class DataLoader:
    def __init__(self, args, train_data, test_data, unseen_data):
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        self.unseen_items = unseen_data
        self.train_items = self.compose_train_items_for_users()

    def compose_train_items_for_users(self):
        train_items = {}
        train_data = self.train_data
        for u_id in range(const.N_USERS):
            u_train_items = train_data[train_data[:, 0] == u_id, 1]
            train_items[u_id] = u_train_items.numpy()
        return train_items

    def get_train_data_by_uid(self, u_id):
        return self.train_data[self.train_data[:, 0] == u_id].clone()

    def get_test_data_by_uid(self, u_id):
        return self.test_data[self.test_data[:, 0] == u_id].clone()

    def get_train_items_by_uid(self, u_id):
        return self.train_items[u_id]

    def get_unseen_items_by_uid(self, u_id):
        return self.unseen_items[u_id]

    def get_test_items_by_uid(self, u_id):
        return self.test_data[self.test_data[:, 0] == u_id, 1].numpy()
        
    def get_user_data_by_iids(self, u_id, i_ids):
        '''
            This function draws the data associated with user ID and a list of item IDs
        '''
        u_data = self.get_train_data_by_uid(u_id)
        indices = np.array([])
        for i_id in i_ids:
            indices = np.concatenate((indices, np.where(self.train_data[:, 1] == i_id)[0]), axis=0)
        return u_data[indices, :-1], u_data[indices, -1]



    def sample_train_data(self, u_ids):
        '''
            This function is specialized for sampling train data for recommendation task in centralized setting
        '''
        # A sample is in the format of (user_id, item_id_1, item_id_2, label),
        # where label = 1 if item_id_1 is seen and item_id_2 is unseen and vice versa
        data = torch.tensor([], dtype=torch.int32).view(-1, 4)
        for u_id in u_ids:
            selected_items = np.random.choice(self.train_items[u_id], size=self.args.n_pos_per_user, replace=True)
            for i_id in selected_items:
                selected_unseen_items = np.random.choice(self.unseen_items[u_id], size=self.args.n_neg_per_pos, replace=True)
                for neg_i_id in selected_unseen_items:
                    new_record = torch.tensor([
                        [u_id, i_id, neg_i_id, 1],
                        #[u_id, neg_i_id, i_id, 0]
                    ])
                    data = torch.cat((data, new_record), 0)
        return Variable(data[:, :-1]).to(self.args.device), Variable(data[:, -1].type(torch.float)).to(self.args.device)
    
    def sample_data_via_uid_and_lr_iid(self, u_id, left_items, right_items, n_right_per_left, is_train):
        '''
            This function creates data for recommendation task in federated setting, where
            each record is comprised of user id, id in left items, id in right items, and the label.
            More precisely, each id in left items is combined with n_right_per_left items in the right items 
            to form the data for the user with u_id
        '''
        data = torch.tensor([], dtype=torch.int32).view(-1, 4)
        # This variable is to keep track of all selected items during the sampling process
        # for further use in requesting item embeddings from the server
        selected_items = [] 
        for i_id in left_items:
            selected_items.append(i_id)
            selected_right_items = np.random.choice(right_items, replace=is_train, size=n_right_per_left)
            selected_items = selected_items + list(selected_right_items)
            # If the left item is a seen/rated item in the train mode or if the client is in the test mode, compose data. Otherwise, no need to compose data for pairs of unseen items
            if (i_id in self.train_items[u_id] and is_train) or (not is_train):
                for neg_i_id in selected_right_items:
                    new_record = torch.tensor([
                        [u_id, i_id, neg_i_id, 1]
                    ])
                    data = torch.cat((data, new_record), 0)

        return Variable(data).to(self.args.device), np.unique(selected_items)

    def sample_single_item_per_record(self, u_id, must_have_items, item_pool, n_items_from_pool):
        '''
            Sample multiple records associated with user with u_id, 
            where each record contains u_id and an item id, 
            and all records must include items in must_have_items & some items randomly selected from the item_pool
            Note that the sampled data will be used for evaluating HR@k and NDCG@k of a model
        '''
        data = torch.tensor([], dtype=torch.int32).view(-1, 2)
        selected_items = np.random.choice(item_pool, replace=False, size=n_items_from_pool)
        for i_id in selected_items:
            data = torch.cat((data, torch.tensor([u_id, i_id]).view(-1, 2)), 0)
        for i_id in must_have_items:
            data = torch.cat((data, torch.tensor([u_id, i_id]).view(-1, 2)), 0)
        return data.to(self.args.device)

    def sample_pair_items_test_data(self, u_ids, n_neg_items):
        '''
            Only sampling positive pairs
            Note that this function is dedicated to sampling test data for recommendation task
            The sampled data is used for evaluating AUC of a model
        '''
        data = torch.tensor([], dtype=torch.int32).view(-1, 3)
        for u_id in u_ids:
            # Since I only extract 1 rated item per user for the test set 
            # => We're only interested in whether this item is preferred over other unrated items
            test_item = self.test_data[u_id, 1]
            for neg_i_id in np.random.choice(self.unseen_items[u_id], size=n_neg_items, replace=False):
                new_record = torch.tensor([u_id, test_item, neg_i_id]).view(-1, 3)
                data = torch.cat((data, new_record), 0)
        return data.to(self.args.device)

    def sample_single_item_test_data(self, u_ids, n_neg_items):
        '''
            This function is dedicated to sampling test data for recommendation task
            The sampled data is used for evaluating HR@k and NDCG@k of a model
        '''
        data = torch.tensor([], dtype=torch.int32).view(-1, 2)
        for u_id in u_ids:
            test_item = self.test_data[u_id, 1]
            # First item in the test sample will be the desired item
            data = torch.cat((data, torch.tensor([u_id, test_item]).view(-1, 2)), 0)
            for neg_i_id in np.random.choice(self.unseen_items[u_id], size=n_neg_items, replace=False):
                new_record = torch.tensor([u_id, neg_i_id]).view(-1, 2)
                data = torch.cat((data, new_record), 0)
        return data.to(self.args.device)




# train_data = torch.load('train_data.pt')
# test_data = torch.load('test_data.pt')

# data_loader = DataLoader(train_data, test_data)
# print('Sampling train data:')
# print(data_loader.sample_train_data([0, 1, 2], 2, 2))
# print('Sampling test data for AUC')
# print(data_loader.AUC_sample_test_data([0, 1, 2]))

