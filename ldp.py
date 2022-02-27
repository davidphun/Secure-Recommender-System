import numpy as np
from const import N_ITEMS
from utils import find_lower_nearest_number
from gurobipy import *

class LocalDP:
    def __init__(self, seen_items):
        self.seen_items = seen_items
        self.p_ratio = 2.5 #
        self.p_seen = 1.0 # Probability to select a seen item during the randomization process
        self.p_unseen = 0.0 # Probability to select an unseen item during the randomization process
        # For some users having very few number of items, the item selection rates need to be optimized
        # Therefore, if the item selection rates of those users have been adjusted, no need to adjust it again
        self.is_optimized = False

    
    def need_optimize(self, n_unseen_in_union, n_neg_per_pos):
        '''
            Check whether the user needs to optimize his privacy
        '''
        p = len(self.seen_items) # number of seen items wrt the user
        q = N_ITEMS - 1 - p # number of unseen items wrt the user
        k = n_neg_per_pos
        kz = float(k / n_unseen_in_union)
        if abs((1/p) - kz) == 0:
        #     # If the denominator of H is less than or equal to 0
            return False
        return True

    def find_optimal_p_seen_and_p_unseen(self, n_unseen_in_union, n_neg_per_pos):
        # Data
        p = len(self.seen_items)
        q = N_ITEMS - 1 - p
        k = n_neg_per_pos
        z = n_unseen_in_union
        a = (1 / q) + (k / z)
        b = (1 / p) - (k / z)
        eps = 2e-2
        if abs(b) < eps:
            b = abs(b)
        H = a / b
        #print(H, a, b)
        t = 1e-2 if b > 0 else a
        # Model
        m = Model('item_selection_rates_optimization')
        m.Params.LogToConsole = 0
        # Variables
        p_seen_ = m.addVar()
        p_unseen_ = m.addVar()
        m.setObjective(p_seen_, GRB.MAXIMIZE)
        # Constraints
        m.addConstr(p_seen_ <= 1.0)
        m.addConstr(p_seen_ >= 0.0)
        m.addConstr(p_unseen_ <= 1.0)
        m.addConstr(p_unseen_ >= 0.0)
        m.addConstr(p_unseen_ == (t / a) + ((1/H) * p_seen_))
        if b > 0:
            m.addConstr(p_unseen_ >= self.p_ratio * p_seen_)
        else:
            m.addConstr(p_seen_ >= p_unseen_)
        # Optimize
        m.optimize()
        p_seen_ = round(p_seen_.x, 2)
        p_unseen_ = round(p_unseen_.x, 2)
        return p_seen_, p_unseen_

    def optimize_item_selection_rates(self, n_unseen_in_union, n_neg_per_pos):
        opt_p_seen, opt_p_unseen = self.find_optimal_p_seen_and_p_unseen(n_unseen_in_union, n_neg_per_pos)
        self.p_seen = opt_p_seen
        self.p_unseen = opt_p_unseen
        self.is_optimized = True
        # print('OPT-p_seen {}, OPT-p_unseen {}'.format(opt_p_seen, opt_p_unseen))
        return
    
    def select_items_via_random_process(self, item_pool):
        selected_items = []
        for i_id in item_pool:
            if i_id in self.seen_items:
                if np.random.rand() < self.p_seen:
                    selected_items.append(i_id)
            else:
                if np.random.rand() < self.p_unseen:
                    selected_items.append(i_id)
        return np.array(selected_items)
