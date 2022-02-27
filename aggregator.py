import numpy as np

class Aggregator:
    def __init__(self):
        self.union_set = np.array([])
        self.aggregated_non_embeddings = {}
        self.aggregated_i_embeddings = {}
        self.aggregated_item_freq = {}
        self.aggregated_n_records = 0
    
    def receive_item_set(self, item_set):
        # Add new item set to the existing union set
        self.union_set = np.concatenate([self.union_set, item_set], axis=0)
        return

    def compose_union_set(self):
        union_set = np.unique(self.union_set).copy()
        # Clear the memory of self.union_set for the use of the next communication...
        self.union_set = np.array([])
        return union_set

    def aggregate_param_data(self, non_embedding_data, i_embedding_data):
        for l, data in non_embedding_data.items():
            if l not in self.aggregated_non_embeddings:
                self.aggregated_non_embeddings[l] = data
            else:
                self.aggregated_non_embeddings[l]['weight'] += data['weight']
                self.aggregated_non_embeddings[l]['bias'] += data['bias']
        for i_id, data in i_embedding_data.items():
            if i_id not in self.aggregated_i_embeddings:
                self.aggregated_i_embeddings[i_id] = data
            else:
                self.aggregated_i_embeddings[i_id] += data
        return

    def aggregate_item_freq_and_n_records(self, item_freq, n_records):
        for i_id, count in item_freq.items():
            if i_id not in self.aggregated_item_freq:
                self.aggregated_item_freq[i_id] = count
            else:
                self.aggregated_item_freq[i_id] += count
        self.aggregated_n_records += n_records
    
    def get_aggregated_data(self):
        aggregated_non_embeddings = self.aggregated_non_embeddings
        aggregated_i_embeddings = self.aggregated_i_embeddings
        # Reset the memory after outputting the aggregated result
        self.aggregated_non_embeddings = {}
        self.aggregated_i_embeddings = {}
        return aggregated_non_embeddings, aggregated_i_embeddings
    
    def get_item_freq_and_n_records(self):
        aggregated_item_freq = self.aggregated_item_freq
        aggregated_n_records = self.aggregated_n_records
        # Reset the memory after outputting the aggregated result
        self.aggregated_item_freq = {}
        self.aggregated_n_records = 0
        return aggregated_item_freq, aggregated_n_records