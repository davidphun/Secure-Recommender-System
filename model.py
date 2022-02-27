import torch
import torch.nn as nn
import torch.nn.functional as F
import const

class SVD_MF(nn.Module):
    def __init__(self, n_latent_dims):
        super(SVD_MF, self).__init__()
        self.n_latent_dims = n_latent_dims
        self.u_embeddings = nn.Embedding(num_embeddings=const.N_USERS, embedding_dim=n_latent_dims)
        self.i_embeddings = nn.Embedding(num_embeddings=const.N_ITEMS, embedding_dim=n_latent_dims)
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.u_embeddings.weight, std=0.01)
        nn.init.normal_(self.i_embeddings.weight, std=0.01)
    
    def forward(self, u_ids, i_ids):
        n_samples = len(u_ids)
        # Reshape the matrix of user embeddings to be of the form 
        # (
        #    u1.T
        #    u2.T
        #    ...
        # )
        u_embeddings = self.u_embeddings(u_ids).view(n_samples, 1, self.n_latent_dims)
        # Reshape the matrix of item embeddings to be of the form
        # ( i1 i2 ... )
        i_embeddings = self.i_embeddings(i_ids).view(n_samples, self.n_latent_dims, 1)
        u_dot_i = torch.bmm(u_embeddings, i_embeddings).view(-1) # Dot product between user embeddings and item embeddings
        return u_dot_i

class NCF(nn.Module):
    def __init__(self, structure):
        super(NCF, self).__init__()
        self.n_latent_dims = structure['n_latent_dims']
        self.n_hidden_layers = structure['n_hidden_layers']
        self.u_embeddings = nn.Embedding(num_embeddings=structure['n_users'], embedding_dim=self.n_latent_dims)
        self.i_embeddings = nn.Embedding(num_embeddings=structure['n_items'], embedding_dim=self.n_latent_dims)
        assert self.n_hidden_layers > 0, 'Cannot create a NCF model with no MLP layer'
        self.mlp_layers = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.mlp_layers.append(nn.Linear(in_features=int(self.n_latent_dims / (2 ** i)), \
                                            out_features=int(self.n_latent_dims / (2 ** (i+1))))
                                    )
        self.output_layer = nn.Linear(in_features=int(self.n_latent_dims / (2 ** self.n_hidden_layers)), out_features=1)
        self.init_weight()
        #print(self.i_embeddings.weight.shape[0])
        #print(self.i_embeddings.weight.data)
    
    def get_mlp_layer_names(self):
        names = [l for l in range(self.n_hidden_layers)]
        names.append('output')
        return names

    def get_mlp_param_data(self, param_type, param_idx, is_grad):
        if param_type == 'output':
            if param_idx == 'weight':
                return self.output_layer.weight.grad.data if is_grad else self.output_layer.weight.data
            else:
                return self.output_layer.bias.grad.data if is_grad else self.output_layer.bias.data
        else:
            if param_idx == 'weight':
                return self.mlp_layers[param_type].weight.grad.data if is_grad else self.mlp_layers[param_type].weight.data
            else:
                return self.mlp_layers[param_type].bias.grad.data if is_grad else self.mlp_layers[param_type].bias.data

    def init_weight(self):
        nn.init.normal_(self.u_embeddings.weight, std=0.01)
        nn.init.normal_(self.i_embeddings.weight, std=0.01)
        for l in self.mlp_layers:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
                if l.bias is not None:
                    l.bias.data.zero_()
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            self.output_layer.bias.data.zero_()

    def forward(self, u_ids, i_ids):
        u_embeddings = self.u_embeddings(u_ids)
        i_embeddings = self.i_embeddings(i_ids)
        mlp_vector = torch.mul(u_embeddings, i_embeddings) # Hadamard product between user and item embeddings
        for l in self.mlp_layers: # Pass the product through multiple MLP layers
            mlp_vector = torch.relu(l(mlp_vector))
        affine_transform = self.output_layer(mlp_vector)
        return affine_transform.view(-1)



# from criterions import BPR
# from torch.autograd import Variable
# import numpy as np
# import copy
# np.random.seed(123)
# torch.manual_seed(123)



# hyperparams = {
#     'n_latent_dims': 128,
#     'n_hidden_layers': 3,
#     'n_users': const.N_USERS,
#     'n_items': const.N_ITEMS
# }
# m = NCF(hyperparams)
# print(m)

# criterion = BPR()
# optimizer = torch.optim.Adam(m.parameters())

# u_ids = Variable(torch.tensor([0, 0, 0, 0, 0]).view(-1, 1))
# l_i_ids = Variable(torch.tensor([1, 1, 1, 2, 2]).view(-1, 1))
# r_i_ids = Variable(torch.tensor([0, 2, 3, 5, 6]).view(-1, 1))
# y = Variable(torch.tensor([1., 1., 1., 1., 1.]))

# # u_ids = Variable(torch.tensor([0]).view(-1, 1))
# # l_i_ids = Variable(torch.tensor([1]).view(-1, 1))
# # r_i_ids = Variable(torch.tensor([3]).view(-1, 1))
# # y = Variable(torch.tensor([1.]))


# # u_ids = Variable(torch.tensor([0]).view(-1, 1))
# # l_i_ids = Variable(torch.tensor([1]).view(-1, 1))
# # r_i_ids = Variable(torch.tensor([2]).view(-1, 1))
# # y = Variable(torch.tensor([1.]))

# # u_ids = Variable(torch.tensor([0]).view(-1, 1))
# # l_i_ids = Variable(torch.tensor([1]).view(-1, 1))
# # r_i_ids = Variable(torch.tensor([0]).view(-1, 1))
# # y = Variable(torch.tensor([1.]))


# optimizer.zero_grad()

# score_l_items = m(u_ids, l_i_ids)
# score_r_items = m(u_ids, r_i_ids)
# loss = criterion(score_l_items, score_r_items, y)
# loss.backward()
# optimizer.step()

# print(m.mlp_layers[0].bias.grad)
# print(m.mlp_layers[0].weight.grad)
# print(m.mlp_layers[0].weight.data.clone())
# print(m.mlp_layers[0].weight.shape)
# print(m.mlp_layers[0].bias.shape)

# for name, param in m.named_parameters():
#     # if name != 'i_embeddings.weight':
#         # continue
#     print(name)
#     print(param.data)
#     print(param.grad)
#     print(param.shape)
#     print('-'*10)



