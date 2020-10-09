from community import community_louvain

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering

import utils
import json
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt




label_dic ={"airplane": 0, 
            "apple": 1, 
            "backpack": 2, 
            "banana": 3, "baseball bat": 4, "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9, "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14, "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19, "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24, "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29, "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34, "hair drier": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39, "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44, "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49, "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54, "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59, "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64, "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69, "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74, "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79}

idx2label = dict((v,k) for k,v in label_dic.items())


def show_graph_with_labels(adjacency_matrix, mylabels, link_type):
    # adjacency_matrix[np.nonzero(adjacency_matrix)]=1
    # print(adjacency_matrix)
    # rows, cols = np.where(adjacency_matrix != 0)
    # edges = zip(rows.tolist(), cols.tolist())
    
    # gr =nx.Graph()
    gr =nx.from_numpy_array(adjacency_matrix)
    epos = [(u, v) for (u, v, d) in gr.edges(data=True) if d["weight"] > 0.05]
    pos = nx.spring_layout(gr)

    for v in gr.nodes():
        gr.nodes[v]['state']=mylabels[v]
    for u,v in gr.edges():
        if (u,v) not in epos:
            gr.remove_edge(u,v)
        else:
            gr[u][v]['type'] = link_type
    gr.remove_nodes_from(list(nx.isolates(gr)))
    print(len(gr.nodes()))

    node_labels = nx.get_node_attributes(gr,'state')
    nx.draw_networkx_labels(gr, pos, labels = node_labels)

    partition = community_louvain.best_partition(gr,weight='weight')
    for node, cluster in dict(partition).items():
        gr.nodes[node]['community'] = cluster
    edges = gr.edges()
    weights = [gr[u][v]['weight'] for u,v in edges]
    #print(gr.edges.data())
    for comm in set(partition.values()):
        key_list=[]

        for key in partition:
            if partition[key] == comm:
                key_list.append(idx2label[key])
            
    
    nx.draw(gr,pos, node_size=50,cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(gr, pos, arrowstyle='->',arrowsize=10,edge_cmap=plt.cm.Greys, width=weights)
    plt.title("best_partition")
    plt.gcf().set_size_inches(16, 13)
    # plt.show()
    return gr


def save_graph_to_json(file_name, pred, to_keep, to_add):
    gr_to_keep = show_graph_with_labels(to_keep, idx2label, 'to_keep')
    if to_add is None:
        gr = gr_to_keep
    else:
        gr_to_add = show_graph_with_labels(to_add, idx2label, 'to_add')
        gr = nx.compose(gr_to_keep, gr_to_add)
    json_data=json_graph.node_link_data(gr)

    del json_data['directed']
    del json_data['multigraph']
    del json_data['graph']

    json_data['predict'] = pred

    for u in range(len(gr.nodes())):
        json_data['nodes'][u].pop('id')
        json_data['nodes'][u]["id"] = json_data['nodes'][u].pop('state')
        json_data['nodes'][u]["group"] = json_data['nodes'][u].pop('community')
        
    for v in range(len(gr.edges())):
        json_data['links'][v]["type"] = json_data['links'][v].pop('type')
        json_data['links'][v]["value"] = json_data['links'][v].pop('weight')
        json_data['links'][v]['source']="".join(idx2label[json_data['links'][v]['source']])
        json_data['links'][v]['target']="".join(idx2label[json_data['links'][v]['target']])
    j = json.dumps(json_data)
    f = open(file_name, 'w')
    f.write(j)
    f.close()

def save_node_adj_to_json(file_name, adj):
    #to_keep: 1, to_add: 2
    label_category = json.load(open('data/coco/labels_category.json','r'))
    label2id = dict((v["name"],v["id"]-1) for v in label_category) 
    valid_ids = {v["id"]-1 for v in label_category}

    #node_list
    gr = show_graph_with_labels(adj, idx2label, 'to_keep')
    json_data=json_graph.node_link_data(gr)

    del json_data['directed']
    del json_data['multigraph']
    del json_data['graph']

    node_reorder = -1.0*np.ones([90,1])
    for u in range(len(gr.nodes())):
        node_idx = label2id[idx2label[u]]
        group = json_data['nodes'][u].pop('community')
        node_reorder[node_idx,0] = group

    node_list = []
    for i in range(node_reorder.shape[0]):
        if i not in valid_ids:
            node_list.append(0)
        else:
            node_list.append({"group": node_reorder[i,0]})

    #edge_list
    adj_reorder = np.zeros([90,90])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            row_idx = label2id[idx2label[i]]
            column_idx = label2id[idx2label[j]]
            adj_reorder[row_idx, column_idx] = adj[i,j]
    
    #save to json
    edge_list = []
    for i in range(adj_reorder.shape[0]):
        if i not in valid_ids:
            edge_list.append(0)
        else:
            tmp_list = []
            for j in range(adj_reorder.shape[1]):
                if adj_reorder[i,j] == 0:
                    tmp_list.append(0)
                else:
                    tmp_list.append([adj_reorder[i,j],1.0])
            edge_list.append(tmp_list)

    json_data = {}
    json_data["links"] = edge_list
    json_data["nodes"] = node_list
    j = json.dumps(json_data)
    f = open(file_name, 'w')
    f.write(j)
    f.close()



def save_adj_to_json(file_name, pred, new_pred, to_keep, to_add):
    #to_keep: 1, to_add: 2
    label_category = json.load(open('data/coco/labels_category.json','r'))
    label2id = dict((v["name"],v["id"]-1) for v in label_category) 
    valid_ids = {v["id"]-1 for v in label_category}

    adj_reorder = np.zeros([90,90])
    ops = np.zeros([90,90])
    for i in range(to_keep.shape[0]):
        for j in range(to_keep.shape[1]):
            row_idx = label2id[idx2label[i]]
            column_idx = label2id[idx2label[j]]
            if to_keep[i,j] > 0.01:
                adj_reorder[row_idx, column_idx] = to_keep[i,j]
                ops[row_idx, column_idx] = 1
            elif to_add[i,j] > 0.01:
                adj_reorder[row_idx, column_idx] = to_add[i,j]
                ops[row_idx, column_idx] = 2
            else:
                adj_reorder[row_idx, column_idx] = 0
    
    #save to json
    edge_list = []
    for i in range(adj_reorder.shape[0]):
        if i not in valid_ids:
            edge_list.append(0)
        else:
            tmp_list = []
            for j in range(adj_reorder.shape[1]):
                if ops[i,j] == 0:
                    tmp_list.append(0)
                else:
                    tmp_list.append([adj_reorder[i,j],ops[i,j]])
            edge_list.append(tmp_list)      
    json_data = {}
    
    json_data["original_predict"] = [{str(label2id[idx2label[k]]):str(v)} for k,v in pred.items()]
    json_data["new_predict"] = []
    for pred in new_pred:
        json_data["new_predict"].append([{str(label2id[idx2label[k]]):str(v)} for k,v in pred.items()])

    json_data["links"] = edge_list
    j = json.dumps(json_data)
    f = open(file_name, 'w')
    f.write(j)
    f.close()





    
   