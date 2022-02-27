from data_loader import DataLoader
from arguments import Arguments
from const import N_USERS, N_ITEMS
import numpy as np
import torch
import pickle
from secure_aggregation.aggregator import Aggregator
from ldp import LocalDP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



class Client:
    def __init__(self, u_id, data_loader, args):
        self.u_id = u_id
        self.data_loader = data_loader
        # Populate the train/test data and unseen/seen items for the client
        self.train_data = data_loader.get_train_data_by_uid(u_id)
        self.test_data = data_loader.get_test_data_by_uid(u_id)
        self.seen_items = data_loader.get_train_items_by_uid(u_id)
        self.unseen_items = data_loader.get_unseen_items_by_uid(u_id)
        self.args = args
        self.item_generator = LocalDP(self.seen_items)
        self.acceptance_rate_for_pos_items = 1.0

    def commit_train_items(self):
        seen_item_set = np.random.choice(self.seen_items, replace=True, size=self.args.n_pos_per_user)
        unseen_item_set = np.random.choice(self.unseen_items, replace=True, size=self.args.n_pos_per_user * self.args.n_neg_per_pos)
        seen_item_set = np.unique(seen_item_set)
        unseen_item_set = np.unique(unseen_item_set)
        # Randomly select/discard some items in the sampled set of positive items...
        seen_item_set = self.select_positive_items_via_randomization(seen_item_set)
        items_to_train = np.concatenate((seen_item_set, unseen_item_set), axis=0)
        return items_to_train
    
    def select_positive_items_via_randomization(self, pre_selected_positive_items):
        # Adjust acceptance rate of positive items 
        # to balance between the update frequency of seen and unseen items...
        self.acceptance_rate_for_pos_items =  min(
            self.args.n_pos_per_user * len(self.seen_items) / len(self.unseen_items),
            1.0
        )

        selected_items = []
        for i_id in pre_selected_positive_items:
            if np.random.rand() < self.acceptance_rate_for_pos_items:
                selected_items.append(i_id)
        return np.array(selected_items)

    def partition_seen_and_unseen_items(self, i_ids):
        seen_items, unseen_items = [], []
        for i_id in i_ids:
            if i_id in self.seen_items:
                seen_items.append(i_id)
            else:
                unseen_items.append(i_id)
        return np.array(seen_items), np.array(unseen_items)

if __name__ == '__main__':
    global_args = Arguments()
    train_data = torch.load('train_data.pt')
    test_data = torch.load('test_data.pt')
    with open('unseen_items.pickle', 'rb') as f:
        unseen_data = pickle.load(f)

    data_loader = DataLoader(global_args, train_data, test_data, unseen_data)

    aggregator = Aggregator()

    clients = []
    for u_id in range(N_USERS):
        args = Arguments()
        # It is more efficient to let all clients and the server have the same model initialization
        # as it is much quicker to converge and more consistent in updating model parameters
        clients.append(Client(u_id, data_loader, args))
        # if len(clients[u_id].seen_items) < 21:
        #     print(u_id)

    examine_u_id = 894 #33, 925, 894, 725, 511, 170, 171, 677, 218, 784, 826
    examine_client = clients[examine_u_id]
    # Keep track of item frequency for each requested item with respect to the selected user
    item_freq = {}
    for t in range(500):
        randomized_items = examine_client.commit_train_items()
        for i_id in randomized_items:
            if i_id not in item_freq:
                item_freq[i_id] = 1
            else:
                item_freq[i_id] += 1

    
    seen_items, unseen_items = examine_client.partition_seen_and_unseen_items(list(item_freq.keys()))
    n_total_seen_items = len(examine_client.seen_items)
    print('N seen items={}, N unseen items={}'.format(n_total_seen_items, N_ITEMS - n_total_seen_items - 1))
    print('Acceptance rate for positive items {}'.format(examine_client.acceptance_rate_for_pos_items))
    print(seen_items)
    for i_id in item_freq.keys():
        if i_id in seen_items:
            print('Id {} is requested {} times'.format(i_id, item_freq[i_id]))
    i_ids = np.array(list(item_freq.keys()))
    freqs = np.array(list(item_freq.values()))
    labels = ['Seen' if i_id in seen_items else 'Unseen' for i_id in i_ids]
    df_plot = pd.DataFrame(data={
        'id': i_ids,
        'freq': freqs,
        'type': labels
    })
    ax = sns.scatterplot(data=df_plot, x='id', y='freq', hue='type')
    ax.set(xlabel='Item ID', ylabel='Frequency')
    ax.set_title('Update/Request frequency of seen and unseen items with respect to user {}\n N_SEEN_ITEMS={}, N_UNSEEN_ITEMS={}'.format(examine_u_id, n_total_seen_items, N_ITEMS - n_total_seen_items - 1))
    plt.show()
