import numpy as np
from numpy.linalg import matrix_power
from utils import largest_indices, get_neighbor_with_k_hops



def evaluate(labels, preds, adj, mask, n=10, k=2):
    """
    num_relevent: number of k-th neighbors for relevent labels
    num_hit: in the top n edges in mask, number of edges that contain true labels
    """
    # if k>1:
        # mask[mask>=0.4]=1
        # mask[mask<0.4]=0
    if k==1:
        kth_mask = mask
        kth_adj = adj
    else:
        kth_mask = get_neighbor_with_k_hops(mask, k)
        
        kth_adj = get_neighbor_with_k_hops(adj, k)
    kth_mask = kth_mask*(1-np.eye(mask.shape[0]))
    kth_adj = kth_adj*(1-np.eye(adj.shape[0]))
    max_index=largest_indices(kth_mask,n)
    labels_set = set(labels)
    preds_set = set(preds)

    num_hit_pred, num_hit_real = 0,0
    num_rel_pred, num_rel_real = 0, 0
    for i in labels:
        for j in labels:
            if kth_adj[i,j] == 1:
                num_rel_real +=1
    for i in preds:
        for j in preds:
            if kth_adj[i,j] == 1:
                num_rel_pred +=1
    for idx in zip(max_index[0],max_index[1]):
        if idx[0] in labels_set and idx[1] in labels_set:
            num_hit_real +=1
        if idx[0] in preds_set and idx[1] in preds_set:
            num_hit_pred +=1
    return num_rel_real, num_rel_pred, num_hit_real, num_hit_pred

def evaluate_faithfulness(scores_, targets_):
    pass

