import torch
import numpy as np
import copy
import math

class Server:
    def __init__(self, model, args):
        self.global_model = copy.deepcopy(model).cpu()
        self.global_args = args
        self.latest_param_square_change = {l:{'weight':1e10, 'bias':1e10} for l in range(model.n_hidden_layers)}
        self.latest_param_square_change['output'] = {'weight':1e10, 'bias':1e10}
        self.latest_param_square_change['item'] = {i_id: 1e10 for i_id in range(model.i_embeddings.weight.shape[0])}
        self.latest_param_square_change['user'] = {u_id: 1e10 for u_id in range(model.u_embeddings.weight.shape[0])}
        self.initialize_parameters_for_Adam_OPT()

    def initialize_parameters_for_Adam_OPT(self):
        self.first_moment_parameters = {}
        self.second_moment_parameters = {}
        # This variable keeps track of how many times a group of params has been updated for 1st & 2nd moment correction in Adam algorithm
        self.param_update_steps = {}
        self.param_update_steps['non-embedding'] = 1
        self.param_update_steps['item'] = {i_id: 1 for i_id in range(self.global_model.i_embeddings.weight.shape[0])}

        for l in range(self.global_model.n_hidden_layers):
            self.first_moment_parameters[l] = {'weight': torch.zeros(self.global_model.mlp_layers[l].weight.shape), 
                                                'bias': torch.zeros(self.global_model.mlp_layers[l].bias.shape)
                                            }
            self.second_moment_parameters[l] = {'weight': torch.zeros(self.global_model.mlp_layers[l].weight.shape), 
                                                'bias': torch.zeros(self.global_model.mlp_layers[l].bias.shape)
                                            }
        self.first_moment_parameters['output'] = {'weight': torch.zeros(self.global_model.output_layer.weight.shape), 
                                                'bias': torch.zeros(self.global_model.output_layer.bias.shape)
                                            }
        
        self.second_moment_parameters['output'] = {'weight': torch.zeros(self.global_model.output_layer.weight.shape), 
                                                'bias': torch.zeros(self.global_model.output_layer.bias.shape)
                                            }
        self.first_moment_parameters['item'] = {i_id: torch.zeros(self.global_model.n_latent_dims) for i_id in range(self.global_model.i_embeddings.weight.shape[0])}
        self.second_moment_parameters['item'] = {i_id: torch.zeros(self.global_model.n_latent_dims) for i_id in range(self.global_model.i_embeddings.weight.shape[0])}
    
    def get_model_param(self, param_type, param_idx):
        if param_type == 'output':
            if param_idx == 'weight':
                return self.global_model.output_layer.weight.data
            else:
                return self.global_model.output_layer.bias.data
        elif param_type == 'item':
            return self.global_model.i_embeddings.weight.data[param_idx]
        else:
            if param_idx == 'weight':
                return self.global_model.mlp_layers[param_type].weight.data
            else:
                return self.global_model.mlp_layers[param_type].bias.data
    
    def update_model_param(self, param_type, param_idx, data):
        if param_type == 'output':
            if param_idx == 'weight':
                self.global_model.output_layer.weight.data = data
            else:
                self.global_model.output_layer.bias.data = data
        elif param_type == 'item':
            self.global_model.i_embeddings.weight.data[param_idx] = data
        else:
            if param_idx == 'weight':
                self.global_model.mlp_layers[param_type].weight.data = data
            else:
                self.global_model.mlp_layers[param_type].bias.data = data
    
    def update_via_FedAvg(self, non_embedding_data, item_embedding_data, item_freq, n_records):
        '''
            Update the global model by Stochastic Gradient Descent method,
            where the structure of the given data is defined in get_parameter_update_data function in client.py file
        '''
        def compute_new_param_data(prev_param_weight, update_data):
            new_param_data = prev_param_weight + update_data
            return new_param_data
        # If no update were made from any client in this communication, abort updating global model
        if n_records == 0:
            return
        # Update parameters for non-embedding component
        for param_type, params in non_embedding_data.items():
            for param_idx, data in params.items():
                old_weight = self.get_model_param(param_type, param_idx)
                aggregated_data = data / n_records if self.global_args.loss_type == 'avg' else data
                new_param_data = compute_new_param_data(self.get_model_param(param_type, param_idx), aggregated_data)
                # This is just to keep track of the latest change of parameters for knowing when to stop training 
                self.store_param_latest_square_change(param_type, param_idx, new_param_data - old_weight)
                self.update_model_param(param_type, param_idx, new_param_data)
        # Update parameters for item-embedding component
        for i_id, data in item_embedding_data.items():
            if item_freq[i_id] == 0:
                # This item does not involve in any update of any client in the current communication.
                # Therefore, it is better to skip updating the global parameters associated with this item
                continue
            old_weight = self.get_model_param('item', i_id)
            aggregated_data = data / item_freq[i_id] if self.global_args.loss_type == 'avg' else data
            new_param_data = compute_new_param_data(self.get_model_param('item', i_id), aggregated_data)
            # This is just to keep track of the latest change of parameters for knowing when to stop training 
            self.store_param_latest_square_change(param_type, param_idx, new_param_data - old_weight)
            self.update_model_param('item', i_id, new_param_data)
        return

    def update_via_FedAdam(self, non_embedding_data, item_embedding_data, item_freq, n_records):
        '''
            Update the global model by Adam method,
            where the structure of the given data is defined in get_parameter_update_data function in client.py file
        '''
        def compute_moments(rate, prev_moments, grad, exponent):
            # regularized_grad = grad - (self.global_args.l2 * prev_weight)
            exp_grad = torch.pow(grad, exponent)
            new_moments = (rate * prev_moments) + ((1-rate) * exp_grad)
            return new_moments

        # If no update were made from any client in this communication, abort updating global model
        if n_records == 0:
            return
        #### Compute the first and second moments for recomputing the weights of the global model
        for param_type, params in non_embedding_data.items():
            for param_idx, data in params.items():
                aggregated_data = data / n_records if self.global_args.loss_type == 'avg' else data
                # Compute the first moments
                self.first_moment_parameters[param_type][param_idx] = compute_moments(
                    self.global_args.momentum, 
                    self.first_moment_parameters[param_type][param_idx], 
                    aggregated_data,
                    1
                )
                # Compute the second moments
                self.second_moment_parameters[param_type][param_idx] = compute_moments(
                    self.global_args.beta2, 
                    self.second_moment_parameters[param_type][param_idx], 
                    aggregated_data,
                    2
                )
        for i_id, data in item_embedding_data.items():
            if item_freq[i_id] == 0:
                continue
            aggregated_data = data / item_freq[i_id] if self.global_args.loss_type == 'avg' else data
            # Compute the first moments
            self.first_moment_parameters['item'][i_id] = compute_moments(
                self.global_args.momentum, 
                self.first_moment_parameters['item'][i_id], 
                aggregated_data,
                1
            )
            # Compute the second moments
            self.second_moment_parameters['item'][i_id] = compute_moments(
                self.global_args.beta2, 
                self.second_moment_parameters['item'][i_id], 
                aggregated_data,
                2
            )
        #### Update global model
        def compute_new_param_data(prev_data, bias_correction_1, bias_correction_2, first_moments, second_moments):
            lr = self.global_args.lr * math.sqrt(bias_correction_2) / bias_correction_1
            eps = self.global_args.eps
            l2 = self.global_args.l2
            return ((1 - l2) * prev_data) - lr * ((first_moments) / (torch.pow(second_moments, 0.5) + eps))
            # return prev_data - lr * ((first_moments) / (torch.pow(second_moments, 0.5) + eps))
        
        ## Update non-embedding parameters
        for param_type, params in non_embedding_data.items():
            for param_idx, _ in params.items():
                old_weight = self.get_model_param(param_type, param_idx)
                new_weight = compute_new_param_data(
                    old_weight,
                    1 - self.global_args.momentum ** self.param_update_steps['non-embedding'],
                    1 - self.global_args.beta2 ** self.param_update_steps['non-embedding'],
                    self.first_moment_parameters[param_type][param_idx],
                    self.second_moment_parameters[param_type][param_idx]
                )
                # This is just to keep track of the latest change of parameters for knowing when to stop training 
                self.store_param_latest_square_change(param_type, param_idx, new_weight - old_weight)
                # Assign new weight to the global model
                self.update_model_param(param_type, param_idx, new_weight)
        
        # Update the step of non-embedding params after 1 training step
        self.param_update_steps['non-embedding'] += 1

        ## Update item-embedding parameters
        for i_id, freq in item_freq.items():
            if freq == 0:
                continue
            old_weight = self.get_model_param('item', i_id)
            new_weight = compute_new_param_data(
                old_weight,
                1 - self.global_args.momentum ** self.param_update_steps['item'][i_id],
                1 - self.global_args.beta2 ** self.param_update_steps['item'][i_id],
                self.first_moment_parameters['item'][i_id],
                self.second_moment_parameters['item'][i_id]
            )
            # This is just to keep track of the latest change of parameters for knowing when to stop training 
            self.store_param_latest_square_change('item', i_id, new_weight - old_weight)
            # Assign new weight to the global model
            self.update_model_param('item', i_id, new_weight)

            # Update the step of item-embedding params after 1 training step
            self.param_update_steps['item'][i_id] += 1
        return

    def get_non_embeddings(self):
        n_hidden_layers = self.global_model.n_hidden_layers
        non_embeddings = {
            l: self.global_model.mlp_layers[l] for l in range(n_hidden_layers)
        }
        non_embeddings['output'] = self.global_model.output_layer
        return non_embeddings

    def get_item_embeddings(self, item_set):
        item_embeddings = {}
        for i_id in item_set:
            item_embeddings[i_id] = self.global_model.i_embeddings.weight.data[i_id]
        return item_embeddings
    
    def store_param_latest_square_change(self, param_type, param_idx, change):
        '''
            Store the latest square aggregated change of parameters for further computing
            the overall change in the global model to determine whether the model converges or not
        '''
        self.latest_param_square_change[param_type][param_idx] = torch.sum(change * change).item()

    def compute_param_sqrt_change(self):
        '''
            Computes the sum of square difference of the weights b4 and after updating
            for all parameters. Then the square root output will be used to determine whether the global
            model converges or not
        '''
        result = 0
        for _, vals in self.latest_param_square_change.items():
            for _, square_change in vals.items():
                result += square_change
        return result ** (1/2)

    def update_model(self, non_embedding_data, i_embedding_data, item_freq, n_records):
        '''
            This function takes non embedding and item embedding data as well as their corresponding frequencies
            to update the global model. In this case, the data could be either the aggregated gradients or changes
            of local parameters.
        '''
        # Here I use FedAdam optimization method to perform update for the global model
        # However, this is a free choice as one can use whatever method to update the global model 
        # as long as it is efficient!
        if self.global_args.fd_update_method == 'FedAdam':
            self.update_via_FedAdam(non_embedding_data, i_embedding_data, item_freq, n_records)
        elif self.global_args.fd_update_method == 'FedAvg':
            self.update_via_FedAvg(non_embedding_data, i_embedding_data, item_freq, n_records)
        return