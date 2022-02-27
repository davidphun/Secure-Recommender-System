# Secure Recommender System

## Introduction

This project is actually my capstone project as a means to complete my Master's thesis at University of Queensland. About the motivation of the project, it was designed to train a recommender system in Federated Learning (FL) setting, where user privacy is protected. More specifically, the proposed solution is dedicated to training a recommender system via Neural Collaborative Filtering (NCF) method and Bayesian Personalized Ranking (BPR) loss while protecting implicit user feedbacks, because these implicit feedbacks are used to train the system, and as many studies shown that knowing implicit feedback data can infer sensitive information about users. Moreover, since the data used to train the recommender system are implicit feedbacks, the model therefore is specialized at recommending which items will likely to be iteracted by a given user in the future. <br><br>

Regarding the high-level detail of the project, the global model in the server is trained by first aggregating the item-embedding gradients as well as the gradients of non-embedding part sent from the clients/users then optimizing the global model parameters with the help of the aggregated gradients via Adam optimizer. In detail, at the beginning of each training iteration, the server selects a subset of users, and these selected users will randomly choose a subset of their seen and unseen items to compose their own training data later, which is called pre-commit item set. Then, these users will submit their pre-commit item sets to construct a union set of items in a way that the server cannot infer which items are submitted by whom thanks to Private Set Union (PSU) algorithm. Afterwards, each user will run a randomization process to select items in the pre-commit item set then sampling items in the union set to compose the training data and finally request the respective item embeddings from the server to train their local recommender systems, where the randomization process is actually the main contribution in this project. To be specific, the reason for introducing the randomization process is to balance the request frequency of every item embedding that a user requested to perform local training, since training a recommender system in BPR approach typically requires the update for seen items more frequent than unseen items if the number of rated items is fewer than the number of unrated items, which is the common case applied to most of the users. Therefore, the server can easily infer which are the interacted items of a particular user based on analysing how frequent the user requests a specific item embedding and knowing the prior knowledge about the user. For example, if the server knows that the user has very few interactions and observes that this user requests item embedding for item i unusually higher than other item embeddings, it is most likely that the user has interacted with this item. As a result, to balance the request frequency, my proposed solution is to run the randomization process to select items from the pre-commit item set as the positive items in the sense of BPR in order to combine with some unseen items sampled from the union set to construct the training dataset, and so the training records composed via this procedure may contain pairs of unseen items, because the randomization process could pick unseen items from the pre-commit item set, which eventually increases the occurances of unseen items in the dataset though these invalid records will not be trained to avoid divergence, but the user will still request the item embeddings of these items to balance the request frequency. For more detail of how the randomization process works and how implicit interactions of users are preserved, please refer to my thesis <a href="https://drive.google.com/file/d/14vuIHMwXK8ZW6we6sz9N1s20jcxQidRX/view">here</a>.

## How to use

- To compose the dataset specialized for BPR approach, one should download MovieLens dataset and modify the path in `prrocess_recommendation_data.py` and run it to construct the training data as well as the test data. <br>

- `args.py` file specifies the main arguments to train a recommender system such as number of training iterations, number of users per communication, and so on. <br>

- Run `main.py` to train the recommender system in either centralized setting or Federated Learning setting. In detail, to train the recommender system in either mode, please check out the example written in the file.<br>

- To visualize how the randomization process balances the request/update frequency of item embeddings, run `item_frequest_analysis.py`. <br>

- To visualize how FedeRank solution can be used to balance the request/update frequency of item embeddings, run `federank_analysis.py`. <br>

- For more detail about the implementation randomization process, please checkout `ldp.py` file. <br>

- To test how the secure aggregation protocol works, run `agg_test.py` in the folder `secure_aggregation`. <br>

- To visualize the experiment results when training the system with various modes, run `performance_comparison.py`. <br>

## Dependencies

Pytorch >= 1.10.0 <br>

PyCryptodome >= 3.14.1 <br>

Matplotlib <br>

## Author

Vy Hoa Phun - 46182856