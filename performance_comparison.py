import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arguments import Arguments

args = Arguments()

metric = 'NDCG@10'

iterations = np.arange(args.log_interval, args.max_iter+1, args.log_interval)
#### Centralized SVD
# AUC
svd_scores = {
    'AUC': [0.82, 0.84, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.89],
    'HR@10': [0.47, 0.49, 0.49, 0.52, 0.53, 0.54, 0.57, 0.57, 0.58, 0.6, 0.6, 0.61, 0.61, 0.63, 0.63],
    'NDCG@10': [0.27, 0.28, 0.28, 0.29, 0.3, 0.31, 0.32, 0.32, 0.32, 0.33, 0.34, 0.35, 0.34, 0.35, 0.36]
}

#### Centralized NCF
# AUC
ncf_scores = {
    'AUC': [0.82, 0.86, 0.87, 0.88, 0.88, 0.89, 0.89, 0.89, 0.89, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    'HR@10': [0.48, 0.58, 0.62, 0.64, 0.63, 0.65, 0.66, 0.66, 0.67, 0.66, 0.67, 0.68, 0.67, 0.68, 0.66],
    'NDCG@10': [0.27, 0.33, 0.35, 0.37, 0.39, 0.38, 0.4, 0.4, 0.41, 0.41, 0.42, 0.42, 0.42, 0.43, 0.42]
}

#### FL NCF by setting p_seen = 1 and p_unseen = 0, i.e., highest risk of privacy disclosure
fl_ncf_scores_full = {
    'AUC': [0.78, 0.85, 0.86, 0.86, 0.87, 0.88, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88],
    'HR@10': [0.49, 0.57, 0.6, 0.61, 0.6, 0.63, 0.63, 0.62, 0.64, 0.65, 0.65, 0.65, 0.63, 0.63, 0.63],
    'NDCG@10': [0.27, 0.31, 0.33, 0.33, 0.34, 0.35, 0.36, 0.36, 0.36, 0.38, 0.38, 0.38, 0.37, 0.37, 0.37]
}

#### FL NCF with optimized p_seen and p_unseen
fl_ncf_scores_optimized = {
    'AUC': [0.51, 0.56, 0.64, 0.72, 0.78, 0.82, 0.84, 0.85, 0.86, 0.86, 0.86, 0.87, 0.87, 0.87, 0.87],
    'HR@10': [0.13, 0.22, 0.33, 0.43, 0.48, 0.5, 0.55, 0.56, 0.57, 0.61, 0.59, 0.6, 0.62, 0.61, 0.62],
    'NDCG@10': [0.06, 0.12, 0.18, 0.24, 0.27, 0.28, 0.31, 0.32, 0.32, 0.34, 0.34, 0.33, 0.35, 0.34, 0.35]
}

df = pd.DataFrame(
    data = {
        'iterations': iterations,
        'Method (1)': svd_scores[metric],
        'Method (2)': ncf_scores[metric],
        'Method (3)': fl_ncf_scores_full[metric],
        'Method (4)': fl_ncf_scores_optimized[metric]
    }
)

df = df.set_index('iterations')

sns.lineplot(data=df)
plt.ylabel(metric)
plt.xlabel('Iterations')
#plt.title('Model performance comparison with respect to {}'.format(metric))
plt.show()


