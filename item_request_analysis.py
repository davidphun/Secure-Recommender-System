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
        # Keep track the indices of seen items during the submission phase to form the union set,
        # which will be used in the randomization process to decide whether the user should those items as the rated items (left items) to perform update
        self.pre_committed_seen_items = None
        self.pre_committed_items = None
        self.zs = []

    def commit_train_items(self):
        seen_item_set = np.random.choice(self.seen_items, replace=True, size=self.args.n_pos_per_user)
        unseen_item_set = np.random.choice(self.unseen_items, replace=True, size=self.args.n_neg_per_commit)
        self.pre_committed_seen_items = seen_item_set
        self.pre_committed_items = np.concatenate([seen_item_set, unseen_item_set], axis=0)
        return self.pre_committed_items.copy()
    
    def compose_randomized_items(self, union_set):
        # Discriminate seen and unseen items in the union set for composing train data
        _, unseen_items = self.partition_seen_and_unseen_items(union_set)
        self.zs.append(len(unseen_items))
        # For some clients with very few seen items, it is most likely that these items will be requested more than frequent
        # and thus just by observing the request frequency, the server should be able to infer which ones are seen items. 
        # Therefore, in order to make the frequency of requesting seen and unseen items indistinguishable

        if not self.item_generator.is_optimized and \
            self.item_generator.need_optimize(len(unseen_items), self.args.n_neg_per_pos):
            #print('UID {} N_SEEN_ITEMS={}-----------------------'.format(self.u_id, len(self.seen_items)))
            self.item_generator.optimize_item_selection_rates(len(unseen_items), self.args.n_neg_per_pos)

        # Get random item indices via randomization process in which 
        # the pre-committed seen items will be paired with a certain number of unseen items in the union set to form the train data
        # randomized_ids = self.item_generator.generate_randomized_items(candidate_l_items)
        
        randomized_ids = self.item_generator.select_items_via_random_process(self.pre_committed_items)

        
        selected_items = []
        for i_id in randomized_ids:
            selected_items.append(i_id)
            selected_right_items = np.random.choice(unseen_items, replace=True, size=self.args.n_neg_per_pos)
            selected_items = selected_items + list(selected_right_items)
        return np.unique(selected_items)

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
        args.n_neg_per_commit = 8
        args.n_pos_per_user = 8
        clients.append(Client(u_id, data_loader, args))
        # if len(clients[u_id].seen_items) < 21:
        #     print(u_id)
    # 111, 226, 777, 592, 896, 33, 925, 894, 725, 511, 170, 171, 677, 218, 784, 826, 456, 526
    examine_u_id = 894

    all_users = np.array([id for id in range(N_USERS) if id != examine_u_id])

    # Keep track of item frequency for each requested item with respect to the selected user
    item_freq = {
        examine_u_id: {}
    }
    for t in range(500):
        selected_users = np.random.choice(all_users, replace=False, size=global_args.n_users_per_batch - 1)
        selected_users = np.concatenate([selected_users, [examine_u_id]], axis=0)
        for u_id in selected_users:
            if u_id not in item_freq:
                item_freq[u_id] = {}
            item_set = clients[u_id].commit_train_items()
            aggregator.receive_item_set(item_set)
        union_set = aggregator.compose_union_set()
        for u_id in selected_users:
            randomized_items = clients[u_id].compose_randomized_items(union_set)
            for i_id in randomized_items:
                if i_id not in item_freq[u_id]:
                    item_freq[u_id][i_id] = 1
                else:
                    item_freq[u_id][i_id] += 1

    examine_client = clients[examine_u_id]
    seen_items, unseen_items = examine_client.partition_seen_and_unseen_items(list(item_freq[examine_u_id].keys()))
    n_total_seen_items = len(examine_client.seen_items)
    print('N seen items={}, N unseen items={}'.format(n_total_seen_items, N_ITEMS - n_total_seen_items))
    print('p_seen {} p_unseen {}'.format(examine_client.item_generator.p_seen, examine_client.item_generator.p_unseen))
    print(seen_items)
    for i_id in item_freq[examine_u_id].keys():
        if i_id in seen_items:
            print('Id {} is requested {} times'.format(i_id, item_freq[examine_u_id][i_id]))
    print('# of unique unseen items: {}'.format(examine_client.zs))
    plt.hist(examine_client.zs)
    plt.show()
    
    #plt.figure(figsize=(10, 10))
    u_item_freq = item_freq[examine_u_id]
    i_ids = np.array(list(u_item_freq.keys()))
    freqs = np.array(list(u_item_freq.values()))
    labels = ['Seen' if i_id in seen_items else 'Unseen' for i_id in i_ids]
    df_plot = pd.DataFrame(data={
        'id': i_ids,
        'freq': freqs,
        'type': labels
    })
    ax = sns.scatterplot(data=df_plot, x='id', y='freq', hue='type')
    ax.set(xlabel='Item ID', ylabel='Frequency')
    ax.set_title('Update/Request frequency of rated and unrated items with respect to user {}\n N_RATED_ITEMS={}, N_UNRATED_ITEMS={}'.format(examine_u_id, n_total_seen_items, N_ITEMS - n_total_seen_items - 1))
    plt.show()
