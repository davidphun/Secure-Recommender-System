import numpy as np
import torch

def auc(y_prob):
    return np.sum(y_prob > 0.5) / len(y_prob)

def hit(desired_item, pred_items):
    if desired_item in pred_items:
        return 1
    return 0

def ndcg(desired_item, pred_items):
    if desired_item in pred_items:
        idx = np.where(pred_items == desired_item)[0][0]
        return np.reciprocal(np.log2(idx + 2))
    return 0

def evaluate_by_auc(model, data_loader, u_ids, n_neg_items):
    model.eval()
    aucs = []
    for u_id in u_ids:
        X_test = data_loader.sample_pair_items_test_data([u_id], n_neg_items)
        score_l_items = model(X_test[:, 0], X_test[:, 1])
        score_r_items = model(X_test[:, 0], X_test[:, 2])
        y_prob = torch.sigmoid(score_l_items - score_r_items).detach().cpu().numpy()
        aucs.append(auc(y_prob))
    return np.mean(aucs)

def evaluate_by_hit_and_ndcg(model, data_loader, u_ids, k, n_neg_items):
    model.eval()
    hits = []
    ndcgs = []
    for u_id in u_ids:
        X_test = data_loader.sample_single_item_test_data([u_id], n_neg_items)
        items = X_test[:, 1]
        desired_item = X_test[0, 1].detach().cpu().item()
        scores = model(X_test[:, 0], items)
        _, indices = torch.topk(scores, k)
        recom_items = torch.take(items, indices).detach().cpu().numpy()
        hits.append(hit(desired_item, recom_items))
        ndcgs.append(ndcg(desired_item, recom_items))
    return np.mean(hits), np.mean(ndcgs)
        
def evaluate_by_auc_fd(users, union_set, n_neg_items):
    aucs = []
    for u in users:
        X_test = u.compose_test_data_for_auc(union_set, n_neg_items)
        score_l_items = u.model(X_test[:, 0], X_test[:, 1])
        score_r_items = u.model(X_test[:, 0], X_test[:, 2])
        y_prob = torch.sigmoid(score_l_items - score_r_items).detach().cpu().numpy()
        aucs.append(auc(y_prob))
    return np.mean(aucs)

def evaluate_by_hit_and_ndcg_fd(users, union_set, k, n_neg_items):
    hits = []
    ndcgs = []
    for u in users:
        X_test, desired_items = u.compose_test_data_for_hit_and_ndcg(union_set, n_neg_items)
        scores = u.model(X_test[:, 0], X_test[:, 1])
        _, indices = torch.topk(scores, k)
        recom_items = torch.take(X_test[:, 1], indices).detach().cpu().numpy()
        hits.append(hit(desired_items, recom_items))
        ndcgs.append(ndcg(desired_items, recom_items))
    return np.mean(hits), np.mean(ndcgs)
