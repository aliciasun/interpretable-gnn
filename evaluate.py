import numpy as np
from numpy.linalg import matrix_power
import utils
from utils import largest_indices, get_neighbor_with_k_hops

label_dic ={"airplane": 0, 
            "apple": 1, 
            "backpack": 2, 
            "banana": 3, "baseball bat": 4, "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9, "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14, "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19, "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24, "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29, "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34, "hair drier": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39, "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44, "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49, "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54, "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59, "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64, "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69, "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74, "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79}

idx2label = dict((v,k) for k,v in label_dic.items())

def evaluate(labels, preds, adj, mask, args, n=10, k=2):
    """
    num_relevent: number of k-th neighbors for relevent labels
    num_hit: in the top n edges in mask, number of edges that contain true labels
    """
    if args.mode == 'promote_v2':
        adj = mask
    if k==1:
        kth_mask = mask
        kth_adj = adj
    else:
        kth_adj = get_neighbor_with_k_hops(adj, k)
        # kth_mask = get_neighbor_with_k_hops(mask, k)
        binary_mask = mask
        binary_mask[binary_mask>=0.4]=1
        binary_mask[binary_mask<0.4]=0
        kth_mask = get_neighbor_with_k_hops(binary_mask, k)
        # kth_binary_mask = get_neighbor_with_k_hops(binary_mask, k)
    
    kth_mask = kth_mask*(1-np.eye(mask.shape[0]))
    kth_adj = kth_adj*(1-np.eye(adj.shape[0]))
    max_index=largest_indices(kth_mask,n)
    labels_set = set(labels)
    preds_set = set(preds)

    num_hit_pred, num_hit_real = 0,0
    num_rel_pred, num_rel_real = 0,0
    num_path_real = 0

    for i in labels:
        for j in labels:
            if kth_adj[i,j] >= 1:
                num_rel_real +=1
                num_path_real += kth_adj[i,j]
    for i in preds:
        for j in preds:
            if kth_adj[i,j] >= 1:
                num_rel_pred +=1
    if args.mode != 'promote_v2':
        for idx in zip(max_index[0],max_index[1]):
            if idx[0] in labels_set and idx[1] in labels_set:
                num_hit_real +=1
            if idx[0] in preds_set and idx[1] in preds_set:
                num_hit_pred +=1
    return num_rel_real, num_rel_pred, num_hit_real, num_hit_pred, num_path_real

def evaluate_faithfulness(scores_, targets_):
    pass

