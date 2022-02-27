from pseudorandom_generator import PNG_AES_CTR
from key_agreement import Diffie_Hellman
from Crypto.Random.random import randint # pip install pycryptodome
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
# Add parent directory to import necessary modules
sys.path.append('..')
from model import NCF
from const import N_ITEMS, N_USERS

class Aggregator:
    def __init__(self, users):
        self.users = users
    
    def generate_user_pairs(self, u_ids):
        pairs = []
        for i, u_id in enumerate(u_ids):
            for v_id in u_ids[i+1:]:
                pairs.append((u_id, v_id))
        return pairs
    
    def exchange_keys(self, update_info):
        for _, u_ids in update_info.items():
            pair_u_ids = self.generate_user_pairs(u_ids)
            for u_id, v_id in pair_u_ids:
                # start_t = time.time()
                self.users[u_id].generate_secret_key(v_id)
                self.users[v_id].generate_secret_key(u_id)
                # print('Time to generate a pair of secret keys: {}'.format(time.time() - start_t))
                if not self.users[v_id].has_peer_pub_key(u_id):
                    # start_t = time.time()
                    # Exchange public keys
                    u_pub_key = self.users[u_id].send_pub_key(v_id)
                    self.users[v_id].save_peer_pub_key(u_id, u_pub_key)
                    v_pub_key = self.users[v_id].send_pub_key(u_id)
                    self.users[u_id].save_peer_pub_key(v_id, v_pub_key)
                    # print('Time to send and store a pair of public keys: {}'.format(time.time() - start_t))
                    # start_t = time.time()
                    # Compute and save shared secret key between u_id and v_id
                    self.users[u_id].obtain_shared_secret_key(v_id)
                    self.users[v_id].obtain_shared_secret_key(u_id)
                    # print('Time to generate shared secret keys: {}'.format(time.time() - start_t))
        return
        
    def aggregate_non_embeddings(self, selected_u_ids):
        n_hidden_layers = self.users[0].model.n_hidden_layers
        aggregated_weight = {l:0 for l in range(n_hidden_layers)}
        aggregated_weight['output'] = 0
        aggregated_bias = {l:0 for l in range(n_hidden_layers)}
        aggregated_bias['output'] = 0
        v_ids = selected_u_ids.copy()
        for u_id in selected_u_ids:
            start_t = time.time()
            for l in range(n_hidden_layers):
                aggregated_weight[l] += self.users[u_id].perturb_non_embedding(v_ids[v_ids != u_id], l, 'weight')
                aggregated_bias[l] += self.users[u_id].perturb_non_embedding(v_ids[v_ids != u_id], l, 'bias')
            aggregated_weight['output'] += self.users[u_id].perturb_non_embedding(v_ids[v_ids != u_id], 'output', 'weight')
            aggregated_bias['output'] += self.users[u_id].perturb_non_embedding(v_ids[v_ids != u_id], 'output', 'bias')
            print('Time to finish computing perturbed weights for 1 user: {}'.format(time.time() - start_t))
        return aggregated_weight, aggregated_bias

    def aggregate_item_embeddings(self, update_info):
        aggregated_result = {}
        for i_id, u_ids in update_info.items():
            v_ids = u_ids.copy()
            aggregated_data = 0
            for u_id in u_ids:
                u_perturbed_item_embedding = self.users[u_id].perturb_item_embedding(i_id, v_ids[v_ids != u_id])
                aggregated_data += u_perturbed_item_embedding
            aggregated_result[i_id] = aggregated_data
        return aggregated_result

class User:
    def __init__(self, u_id, model):
        self.u_id = u_id
        self.dh_exchange = Diffie_Hellman()
        self.secret_keys = {}
        self.model = model
        self.peer_pub_keys = {}
        self.embed_pngs = {}
        self.shared_secret_keys = {}
        self.non_embed_pngs = {}
    
    def has_peer_pub_key(self, peer_u_id):
        return peer_u_id in self.peer_pub_keys
    
    def save_peer_pub_key(self, u_id, key):
        self.peer_pub_keys[u_id] = key

    def generate_secret_key(self, peer_u_id):
        if peer_u_id not in self.secret_keys:
            # self.secret_keys[peer_u_id] = randint(2**255, self.dh_exchange.prime - 1) # The minimum key length recommended for 2048-bit MODP to meet the security level
            self.secret_keys[peer_u_id] = randint(1, self.dh_exchange.prime - 1)
    def send_pub_key(self, peer_u_id):
        # assert peer_u_id in self.secret_keys, 'No keys were generated for User {}'.format(peer_u_id)
        # assert i_id in self.secret_keys[peer_u_id], 'Secret key for {} has not been generated!'.format(i_id)
        return self.dh_exchange.generate_public_key(self.secret_keys[peer_u_id])

    def obtain_shared_secret_key(self, peer_u_id):
        if peer_u_id not in self.shared_secret_keys:
            shared_secret_key = self.dh_exchange.compute_shared_secret_key(
                self.peer_pub_keys[peer_u_id],
                self.secret_keys[peer_u_id]
            )
            trimmed_shared_secret_key = shared_secret_key >> (shared_secret_key.bit_length() - (16 * 8))
            self.shared_secret_keys[peer_u_id] = trimmed_shared_secret_key

    def perturb_item_embedding(self, i_id, peer_u_ids):
        vec_size = self.model.i_embeddings.weight.data[i_id, :].shape[0]
        perturbed_vector = self.model.i_embeddings.weight.data[i_id, :].numpy()
        for peer_u_id in peer_u_ids:
            # Initialize PNGs if not exist
            if peer_u_id not in self.embed_pngs:
                self.embed_pngs[peer_u_id] = {}
            if i_id not in self.embed_pngs[peer_u_id]:
                self.embed_pngs[peer_u_id][i_id] = PNG_AES_CTR(self.shared_secret_keys[peer_u_id])
            # Perturb embedding weights
            if peer_u_id < self.u_id:
                perturbed_vector += self.embed_pngs[peer_u_id][i_id].random_std_normal_ziggurat(vec_size)
            else:
                perturbed_vector -= self.embed_pngs[peer_u_id][i_id].random_std_normal_ziggurat(vec_size)
        return perturbed_vector

    def perturb_non_embedding(self, peer_u_ids, layer_type, data_type):
        # Initialize PNGs if not exist
        for peer_u_id in peer_u_ids:
            if peer_u_id not in self.non_embed_pngs:
                self.non_embed_pngs[peer_u_id] = PNG_AES_CTR(self.shared_secret_keys[peer_u_id])
        # Obtain non-embedding weights/biasses
        perturbed_vector = None
        if data_type == 'bias':
            perturbed_vector = self.model.output_layer.bias.data.numpy() if layer_type == 'output' else \
                                self.model.mlp_layers[layer_type].bias.data.numpy()
        else:
            perturbed_vector = self.model.output_layer.weight.data.numpy() if layer_type == 'output' else \
                                self.model.mlp_layers[layer_type].weight.data.numpy()
        vec_shape = tuple(perturbed_vector.shape)
        # Perturb non-embedding weights/bias
        start_t = time.time()
        for peer_u_id in peer_u_ids:
            if peer_u_id < self.u_id:
                perturbed_vector += self.non_embed_pngs[peer_u_id].random_std_normal_ziggurat(vec_shape)
            else:
                perturbed_vector -= self.non_embed_pngs[peer_u_id].random_std_normal_ziggurat(vec_shape)
        print('Time to perturb 1 vector with shape {}: {}'.format(vec_shape, time.time() - start_t))
        return perturbed_vector

n_selected_users = 20

hyperparams = {
    'n_latent_dims': 128,
    'n_hidden_layers': 3,
    'n_users': N_USERS,
    'n_items': N_ITEMS
}

users = {
    i: User(i, NCF(hyperparams)) for i in range(n_selected_users)
}



agg = Aggregator(users)

update_info = {
    0: np.array([i for i in range(n_selected_users)])
}
start_t = time.time()
agg.exchange_keys(update_info)
print('Key Exchange time: {}'.format(time.time() - start_t))
start_t = time.time()
aggregated_embeddings = agg.aggregate_item_embeddings(update_info)
aggregated_mlp_weight, aggregated_mlp_bias = agg.aggregate_non_embeddings(np.array([i for i in range(n_selected_users)]))
print('Aggregation time: {}'.format(time.time() - start_t))
# print(aggregated_embeddings[0])
print(aggregated_mlp_weight[0])
print(aggregated_mlp_bias[0])
# true_result = sum([users[i].model.i_embeddings.weight.data[0, :] for i in range(n_selected_users)])
# print(true_result)
# print(aggregated_embeddings[0] == true_result)
true_weight = sum([users[i].model.mlp_layers[0].weight.data for i in range(n_selected_users)])
true_bias = sum([users[i].model.mlp_layers[0].bias.data for i in range(n_selected_users)])
print(true_weight)
print(true_bias)

# dh = Diffie_Hellman()

# secret_key1 = randint(2**255, dh.prime - 1)
# secret_key2 =  randint(2**255, dh.prime - 1)

# pub_key1 = dh.generate_public_key(secret_key1)
# pub_key2 = dh.generate_public_key(secret_key2)

# print('Public key for secret key {}: {}'.format(secret_key1, pub_key1))
# print('Public key for secret key {}: {}'.format(secret_key2, pub_key2))
# print('Shared secret key: {}'.format(dh.compute_shared_secret_key(pub_key1, secret_key2)))
# print('Shared secret key: {}'.format(dh.compute_shared_secret_key(pub_key2, secret_key1)))

# shared_secret_key = dh.compute_shared_secret_key(pub_key1, secret_key2)
# print(shared_secret_key.bit_length())
# trimmed_shared_secret_key = shared_secret_key >> (shared_secret_key.bit_length() - (16 * 8))
# print(trimmed_shared_secret_key)
# print('Trimmed 128-bit shared secret key: {}'.format(trimmed_shared_secret_key.to_bytes(16, 'little')))

# png = PNG_AES_CTR(trimmed_shared_secret_key)


# plt.hist(png.random_std_normal(10000))
# plt.show()


