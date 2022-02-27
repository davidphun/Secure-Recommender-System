import torch
import torch.nn.functional as F
import numpy as np
import const
import pickle

class DataPreprocessor:
    def __init__(self):
        return

    @staticmethod
    def load_data_from_file(file_name):
        data = []
        with open(file_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            tokens = line.split('\t')
            data.append([int(tokens[0])-1, int(tokens[1])-1, int(tokens[3])])
        return torch.tensor(data)
    
    @staticmethod
    def split_train_test_set(data):
        u_ids = data[:, 0].unique()
        train_data, test_data = torch.tensor([], dtype=torch.int32).view(-1, 2), torch.tensor([], dtype=torch.int32).view(-1, 2)
        for u_id in u_ids:
            u_data = data[data[:, 0] == u_id]
            u_data_sorted = u_data[np.argsort(-1 * u_data[:, 2])] # Sort user data accordance with the timestamp in descending order
            # Get the latest rated movie from a user to create data for testing set
            hold_out_item = u_data_sorted[0, 1]
            u_train = u_data[u_data[:, 1] != hold_out_item, :-1]
            u_test = u_data[u_data[:, 1] == hold_out_item, :-1]
            train_data = torch.cat((train_data, u_train), 0) 
            test_data = torch.cat((test_data, u_test), 0)
        return train_data, test_data
    @staticmethod
    def compose_unseen_items_for_users(train_data, test_data):
        unseen_items = {}
        train_data = train_data
        test_data = test_data
        for u_id in range(const.N_USERS):
            train_items = train_data[train_data[:, 0] == u_id, 1] # All items rated by user in the train set
            test_items = test_data[test_data[:, 0] == u_id, 1] # All items rated by user in the test set
            u_unseen_items = np.array([i for i in range(const.N_ITEMS) if (i not in train_items) and (i not in test_items)])
            unseen_items[u_id] = u_unseen_items
        return unseen_items

ml_data = DataPreprocessor.load_data_from_file('./ml-100k/u.data')

train_data, test_data = DataPreprocessor.split_train_test_set(ml_data)
unseen_items = DataPreprocessor.compose_unseen_items_for_users(train_data, test_data)

print(train_data[train_data[:, 0] == 0])
print(test_data[test_data[:, 0] == 0])

torch.save(train_data, 'train_data.pt')
torch.save(test_data, 'test_data.pt')

with open('unseen_items.pickle', 'wb') as f:
    pickle.dump(unseen_items, f, protocol=pickle.HIGHEST_PROTOCOL)
