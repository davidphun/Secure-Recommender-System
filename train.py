from arguments import Arguments
import const
from evaluation import evaluate_by_auc, evaluate_by_hit_and_ndcg, evaluate_by_auc_fd, evaluate_by_hit_and_ndcg_fd
import numpy as np
import torch
import time
from criterions import BPR
from client import Client
from aggregator import Aggregator
from server import Server
import copy


def centralized_train(model, data_loader, args):
    all_users = np.arange(const.N_USERS)
    criterion = BPR()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.beta2), eps=args.eps, weight_decay=args.l2)
    for i in range(args.max_iter):
        #print('Iter {}'.format(i))
        selected_users = np.random.choice(all_users, size=args.n_users_per_iter, replace=False)
        model.train()
        X, y = data_loader.sample_train_data(selected_users, None)            
        optimizer.zero_grad()
        score_l_items = model(X[:, 0], X[:, 1])
        score_r_items = model(X[:, 0], X[:, 2])
        loss = criterion(score_l_items, score_r_items, y)
        loss.backward()
        optimizer.step()
        if (i+1) % args.log_interval == 0:
            print('-'*10)
            print('Iteration {}'.format(i+1))
            avg_auc = evaluate_by_auc(model, data_loader, all_users, 100)
            avg_hit, avg_ndcg = evaluate_by_hit_and_ndcg(model, data_loader, all_users, 10, 100)
            print('Neg LLH Loss: {}, Average AUC: {}'.format(round(loss.item(), 2), round(avg_auc, 2)))
            print('HR@{}:{}, NDCG@{}:{}'.format(10, round(avg_hit, 2), 10, round(avg_ndcg, 2)))
        grads = torch.cat([param.grad.view(-1) for param in model.parameters()])
        squared_grads = torch.pow(grads, 2)
        sum_sqrt_grads = torch.pow(torch.sum(squared_grads), 1/2).detach().cpu().item()
        if sum_sqrt_grads <= args.eps: # If the gradients are extremely small, stop the training process
            print('Training stops at iteration {}'.format(i))
            break
    return model

def federated_train(global_model, data_loader, global_args):
    server = Server(global_model, global_args)
    aggregator = Aggregator()
    clients = []
    #### Initialize clients
    print('Initialize clients')
    for u_id in range(const.N_USERS):
        args = Arguments()
        # It is more efficient to let all clients and the server have the same model initialization
        # as it is much quicker to converge and more consistent in updating model parameters
        clients.append(Client(u_id, data_loader, global_model, args))
    print('GPU memory reserved: {}'.format(torch.cuda.memory_reserved(0)))
    print('GPU memory allocated: {}'.format(torch.cuda.memory_allocated(0)))
    print('Perform training')
    #### Perform training
    T = global_args.max_iter # Number of communications
    client_indices = np.arange(const.N_USERS)
    for t in range(T):
        # print(t)
        # print('GPU memory reserved: {}'.format(torch.cuda.memory_reserved(0)))
        # print('GPU memory allocated: {}'.format(torch.cuda.memory_allocated(0)))
        selected_clients = np.random.choice(client_indices, replace=False, size=global_args.n_users_per_iter)
        ## Commit item to form union set
        for u_id in selected_clients:
            aggregator.receive_item_set(clients[u_id].commit_train_items())
        ## Perform training and updating global model...
        # Compose the union set
        union_set = aggregator.compose_union_set().astype(int)
        for u_id in selected_clients:
            # Pass the union set to the selected client for randomly picking items 
            # to compose train data in the local training process...
            data, selected_items = clients[u_id].compose_train_data_via_randomized_items(union_set)
            # Request necessary global parameters from the server
            non_embeddings = server.get_non_embeddings()
            i_embeddings = server.get_item_embeddings(selected_items)
            # Save the requested global parameters to the local model in order to perform local update
            clients[u_id].save_global_parameters(non_embeddings, i_embeddings)
            # Perform local training
            clients[u_id].train_recommendation(data)
            # Deallocate data after training
            del data
            torch.cuda.empty_cache()
            # Compute the grad/change in parameters after training
            non_embedding_data, i_embedding_data, u_embedding_sum_square_change = clients[u_id].get_parameter_update_data(non_embeddings, i_embeddings)
            # Get item freq and # of records used for the training
            item_freq, n_records = clients[u_id].get_item_freq_and_n_records()
            # Submit the change as well as the weight of each chosen item to perform aggregation
            aggregator.aggregate_param_data(non_embedding_data, i_embedding_data)
            aggregator.aggregate_item_freq_and_n_records(item_freq, n_records)
            # Submit the sum of square change of the user embedding to the server
            # Note that this step is for evaluating the overall change in the global model in the later step
            server.store_param_latest_square_change('user', u_id, u_embedding_sum_square_change ** (1/2))
        ## Aggregate to update the global parameters
        non_embedding_data, i_embedding_data = aggregator.get_aggregated_data()
        item_freq, n_records = aggregator.get_item_freq_and_n_records()
        server.update_model(non_embedding_data, i_embedding_data, item_freq, n_records)
        ## Evaluate the square root change of global model parameters
        sqrt_param_change = server.compute_param_sqrt_change()
        if sqrt_param_change <= args.eps:
            print('Training stop at communication {}'.format(t))
            break
        ## Evaluate the model performance after several iterations
        if (t+1) % global_args.log_interval == 0:
            print('-'*10)
            print('Iteration {}'.format(t+1))
            print(sqrt_param_change)
            #### Real evaluation process... but this one takes very long to actually evaluate the global model performance
            # In this scenario, I'm interested in the true model performance and thus I want to evaluate
            # how the model performs among ALL users. In practice, the server can only select a subset of users
            # to measure the model performance due to the expensive communication cost
            # selected_users = np.arange(const.N_USERS)
            # for u_id in selected_users:
            #     ## Commit item to form union set
            #     aggregator.receive_item_set(clients[u_id].commit_test_items(10))
            # # Compose the union set
            # union_set = aggregator.compose_union_set().astype(int)
            # for u_id in selected_users:
            #     # Get global parameters from server
            #     non_embeddings, i_embeddings = server.get_non_embeddings(), server.get_item_embeddings(union_set)
            #     # Save the requested global parameters to the local model to perform evaluation
            #     clients[u_id].save_global_parameters(non_embeddings, i_embeddings)
            # selected_clients = [clients[u_id] for u_id in selected_users]
            # auc = evaluate_by_auc_fd(selected_clients, union_set, 100)
            # hit, ndcg = evaluate_by_hit_and_ndcg_fd(selected_clients, union_set, 10, 100)
            # print('Average AUC: {}'.format(round(auc, 2)))
            # print('HR@{}:{}, NDCG@{}:{}'.format(10, round(hit, 2), 10, round(ndcg, 2)))

            #### Novel evaluation process... this is fast for debugging 
            # Let all users submit their user embeddings to the server and assume the server has all user data
            # => The evaluation process from now on is identical to that in centralized setting
            all_users = np.arange(const.N_USERS)
            global_model = copy.deepcopy(server.global_model).to(global_args.device)
            for u_id in all_users:
                global_model.u_embeddings.weight.data[u_id] = clients[u_id].model.u_embeddings.weight.data[u_id]

            avg_auc = evaluate_by_auc(global_model, data_loader, all_users, 100)
            avg_hit, avg_ndcg = evaluate_by_hit_and_ndcg(global_model, data_loader, all_users, 10, 100)
            print('Average AUC: {}'.format(round(avg_auc, 2)))
            print('HR@{}:{}, NDCG@{}:{}'.format(10, round(avg_hit, 2), 10, round(avg_ndcg, 2)))