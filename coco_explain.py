#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import math
import heapq
import shutil
import torch
import time
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch.optim
import torch.nn as nn
import torchnet as tnt
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

import utils
import utils_viz
from dataset import CocoGender, COCO2014
from models import gcn_resnet101, Warp, gen_A
from interpreter import GNNInterpreter, smooth_adj
from evaluate import evaluate, evaluate_faithfulness
import statistics




label_dic ={"airplane": 0, 
            "apple": 1, 
            "backpack": 2, 
            "banana": 3, "baseball bat": 4, "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9, "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14, "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19, "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24, "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29, "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34, "hair drier": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39, "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44, "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49, "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54, "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59, "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64, "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69, "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74, "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79}

idx2label = dict((v,k) for k,v in label_dic.items())

def explain_group(model, val_loader, orig_A, target_label, subgroup, args, method = 'grad'):
  
    use_cuda = torch.cuda.is_available()
    print("cuda: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    args.mode = 'group'
    adj=torch.Tensor(orig_A)
    interpreter = GNNInterpreter(model, adj, None, None, None, None, args)
    for step in tqdm(range(args.num_steps)):
        interpreter.zero_grad()
        interpreter.optimizer.zero_grad()
        if interpreter.adj.grad is not None:
            interpreter.adj.grad.zero_()
            interpreter.x.grad.zero_()
            interpreter.mask.grad.zero_()
        batch_size = 32
        for i, (inp, target) in enumerate(val_loader):
            photo=Variable(inp[0], requires_grad=True).float().to(device)
            feature=Variable(inp[3], requires_grad=True).float().to(device)
            orig_pred=model(photo, feature)
            orig_pred_prob = torch.sigmoid(orig_pred)
            cur_pred, cur_pred_prob = interpreter(photo, feature)
            loss = nn.MultiLabelSoftMarginLoss()(cur_pred, orig_pred_prob)
            avg_loss = torch.mean(loss)
            avg_loss = avg_loss - 0.001*torch.norm(interpreter.mask_existing, p=1)
            avg_loss.backward(retain_graph=True)
            interpreter.mask.grad[interpreter.budget>0]=0            
            interpreter.mask.grad.data.fill_diagonal_(0)
            interpreter.optimizer.step()
            mask_existing = interpreter.mask_existing
            mask_to_add = interpreter.mask_to_add
            interpreter.exp_lr_scheduler.step()
        if step%10 == 0:
            print('[iter:%d] to_remove mask weight: %.3f to_add mask weight: %.3f loss: %.3f' %
            (step, torch.norm(mask_existing, p=1), torch.norm(mask_to_add, p=1),torch.sum(loss)))
    mask_to_add = mask_to_add.detach().numpy()*(1-np.eye(orig_A.shape[0]))
    mask_to_keep = (orig_A-mask_existing.detach().numpy())*(1-np.eye(orig_A.shape[0]))
    
    print(utils.get_top_k_pairs(mask_to_keep, idx2label, k=10), sep = "\n")
    print('-------------------------------------------')
    with open('gender/pred_{0}_{1}.npz'.format(target_label,subgroup), 'wb') as f:
        np.savez(f, to_keep = mask_to_keep, to_add = mask_to_add)
    f.close()



def explain(model, val_loader, orig_A, args, method = 'mask'):
    use_cuda = torch.cuda.is_available()
    print("cuda: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    hit_pred, hit_real = [], []
    perf_pred, perf_real = [], []
    total_rel_pred, total_rel_real, total_hit_pred, total_hit_real, total_performance, total_performance_real =0, 0, 0, 0, 0, 0
    total_performance_add, total_performance_real_add = 0,0
    valid_sample, valid_sample_real = 0, 0
    common_pred = []
    for i, (inp, target) in enumerate(val_loader):
        print("training for image: {0}".format(i))
        # if i > 50:
        #     break
        photo=Variable(inp[0], requires_grad=True).float()
        img_path = inp[1][0].split(".")[0]
           
        if args.dataset == 'coco':
            feature=Variable(inp[2], requires_grad=True).float()
        else:
            feature=Variable(inp[3], requires_grad=True).float()
        label_idx = np.where(target==1)[1]
        true_labels = [idx2label[l] for l in label_idx]
        
        true_label_length = len(true_labels)
        print("ground truth label....",true_labels)

        pred=model(photo, feature)
        orig_pred_prob = torch.sigmoid(pred)
        pred_list = torch.sigmoid(pred[0]).detach().numpy()
        preds = list(pred_list.argsort()[-true_label_length:][::-1])
        pred_prob={i:pred_list[i] for i in preds}
        predicted_labels = [idx2label[l] for l in preds]

        print("predict label.....",predicted_labels)
        # if args.mode =='target_attack' and args.target_label in label_idx:
        #     continue

        if method == 'grad':
            pred = torch.sigmoid(pred)
            pred.backward(gradient=torch.ones_like(pred),retain_graph=True)
            grad_A=model.A.grad.detach()
            grad_A = grad_A.numpy()
            pos_grad = grad_A*(grad_A>=0)
            neg_grad = grad_A*(grad_A<0)

            mask_to_keep = pos_grad*orig_A
            mask_to_add = pos_grad*(1.0-orig_A)
            # file_name = 'subgraph_coco_gender/image/{}.json'.format(img_path)
            # utils_viz.save_adj_to_json(file_name, pred_prob, to_keep, to_add)
        elif method == 'mask':
            adj=torch.Tensor(orig_A)
            interpreter = GNNInterpreter(model, adj, photo, feature, target, pred, args)

            for step in tqdm(range(args.num_steps)):
                interpreter.zero_grad()
                interpreter.optimizer.zero_grad()
                if interpreter.adj.grad is not None:
                    interpreter.adj.grad.zero_()
                    interpreter.x.grad.zero_()
                    interpreter.mask.grad.zero_()
                cur_pred, cur_pred_prob = interpreter(photo, feature)
                loss = interpreter.loss(cur_pred, orig_pred_prob)
                loss.backward(retain_graph=True)
                if args.mode == 'preserve':
                    interpreter.mask.grad[interpreter.budget>0]=0
                interpreter.mask.grad.data.fill_diagonal_(0)
                # interpreter.mask.grad[torch.eye(adj.shape[0]).bool()]=0
                # causal_edges = torch.transpose(interpreter.budget<0, 0,1)
                # interpreter.mask.grad[causal_edges]=0

                interpreter.optimizer.step()
                if args.mode == 'preserve':
                    mask_to_add = interpreter.mask_to_add
                else:
                    mask_existing = interpreter.mask*(interpreter.budget<0)
                    mask_to_add = interpreter.mask*(interpreter.budget>0)
                interpreter.exp_lr_scheduler.step()
                if step%10 == 0:
                    print('[iter:%d] to_remove mask weight: %.3f to_add mask weight: %.3f loss: %.3f' %
                    (step, torch.norm(mask_existing, p=1), torch.norm(mask_to_add, p=1),torch.sum(loss)))
            mask_to_add = mask_to_add.detach().numpy()*(1-np.eye(orig_A.shape[0]))
            mask_to_keep = (orig_A-mask_existing.detach().numpy())*(1-np.eye(orig_A.shape[0]))
            if args.mode == 'preserve':
                masked_adj_smooth = smooth_adj(interpreter.masked_adj.detach())
            else:
                masked_adj_smooth = interpreter.masked_adj.detach()
            new_pred=model(photo, feature, adj = masked_adj_smooth)
            new_pred = torch.sigmoid(new_pred)
            new_pred_list = new_pred[0].detach().numpy()
            new_preds = list(new_pred_list.argsort()[-true_label_length:][::-1])

            new_pred_prob={i:new_pred_list[i] for i in new_preds}
            new_predicted_labels = [idx2label[l] for l in new_preds]
            print("predict label.....",new_predicted_labels)
            
            common_object = len(list(set(preds).intersection(new_preds)))
            common_pred.append(common_object/len(preds))
            # common_pred_rank.append()

            # file_name = 'mask_interpreter/json/{}/{}.json'.format(args.mode,img_path)
            # utils_viz.save_adj_to_json(file_name, pred_prob, new_pred_prob, mask_to_keep, mask_to_add)
            print(*utils.get_top_k_pairs(mask_to_keep, idx2label, k=10), sep = "\n")
            # print('-------------------------------------------')
            # print(*utils.get_top_k_pairs(mask_to_add, idx2label, k=10), sep = "\n")
            # mask_to_keep_k_hop = utils.get_neighbor_with_k_hops(mask_to_keep,2)*(1-np.eye(orig_A.shape[0]))

            # with open('mask_interpreter/mask/{}/{}.npz'.format(args.mode, img_path), 'wb') as f:
            #     np.savez(f, to_keep = mask_to_keep, to_add =mask_to_add)
            # print(*utils.get_top_k_pairs(mask_to_keep_k_hop, idx2label, k=10), sep = "\n")
            # print('-------------------------------------------')
            # print(*utils.get_top_k_pairs(mask_to_add, idx2label, k=10), sep = "\n")

            # print('-------------------------------------------')
            # print(*utils.get_top_k_pairs(mask_to_add, idx2label),sep = "\n")
        print("common pred: {0}".format(statistics.mean(common_pred)))
        # print("common pred rank: {0}".format(statistics.mean(common_pred_rank)))

        for k in [1,2]:
            print("evaluating on {0}-hop neighbors".format(k))
            num_rel_real, num_rel_pred, num_hit_real, num_hit_pred = evaluate(label_idx, preds, orig_A, mask_to_keep, k=k)
            
            # total_rel += num_rel_pred
            # total_hit += num_hit_pred
            # total_hit_real += num_hit_real
            if num_rel_pred != 0 :
                hit_pred.append(num_hit_pred)
                perf_pred.append(num_hit_pred/num_rel_pred)
                # total_performance += num_hit_pred/num_rel_pred
                # total_performance_add += num_hit_add/num_rel_pred
                # valid_sample += 1 
            if num_rel_real != 0:
                hit_real.append(num_hit_real)
                perf_real.append(num_hit_real/num_rel_real)
                # total_performance_real += num_hit_real/num_rel_real
                # total_performance_real_add += num_hit_real_add/num_rel_real
                # valid_sample_real += 1 
            

            print("number of relevent predict edges: {0}".format(num_rel_pred))
            print("number of relevent real edges: {0}".format(num_rel_real))
            print("number of hit predict edges: {0}".format(num_hit_pred))
            print("number of hit real edges: {0}".format(num_hit_real))
            print("average performance pred: {0}".format(statistics.mean(perf_pred)))
            print("average performance real: {0}".format(statistics.mean(perf_real)))
           
            if args.mode != 'preserve':
                _, num_rel_add, num_hit_real_add, num_hit_add = evaluate(label_idx, preds, orig_A, mask_to_keep+mask_to_add, k=k)
                print("number of hit predict edges after add: {0}".format(num_hit_add))
                print("number of hit real edges after add: {0}".format(num_hit_real_add))
                print("average performance real after add: {0}".format(total_performance_real_add/valid_sample_real))
                print("average performance after add: {0}".format(total_performance_add/valid_sample))


        


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="dataset"
    )
    parser.add_argument(
        "--method", dest="method", type=str, help="Number of epochs to train."
    )
    parser.add_argument(
        "--mode", dest="mode", type=str, help="Number of epochs to train."
    )
    parser.add_argument(
        "--steps", dest="num_steps", type=int, help="Number of iters to train the mask."
    )
    parser.add_argument(
        "--target_label", dest="target_label", type=int, help="attack target label"
    )
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.set_defaults(
        dataset='coco',
        method='mask',
        mode='preserve',
        num_steps=10,
        target_label=49 # person:49 dog:28
    )
    args = parser.parse_args()

    adj=gen_A(num_classes=80,t=0.4, adj_file='data/coco/coco_adj.pkl', p=1)

    # adj[np.nonzero(adj)]=1  

    #use pretrained ML_GCNmodel
            
    model = gcn_resnet101(num_classes=80, t=0.4, adj_file = 'data/coco/coco_adj.pkl', interpret_mode = True)
    checkpoint=torch.load('coco_checkpoint.pth.tar',map_location=torch.device('cpu'))

                        
    model.load_state_dict(checkpoint['state_dict'],False)                     
    model.eval()
    if args.dataset == 'coco':
        val_dataset = COCO2014(root='data/coco', phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    else:
        val_dataset = CocoGender(phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                                std=model.image_normalization_std)
    val_dataset.transform = transforms.Compose([
                Warp(448),
                transforms.ToTensor(),
                normalize,
            ])
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    if args.mode == 'group':

    # file_name = 'subgraph_coco_gender/ground_truth.json'
    # utils_viz.save_node_adj_to_json(file_name, adj)
        target_labels =  ['tennis racket','knife','motorcycle']
        label_dic_female = dict.fromkeys(target_labels, []) 
        label_dic_male = dict.fromkeys(target_labels, []) 
        for i, (inp, target) in enumerate(val_loader):
            gender = Variable(inp[2], requires_grad=False).detach().numpy()
            label_idx = np.where(target==1)[1]
            true_labels = [idx2label[l] for l in label_idx]
            for t in target_labels:
                if t in true_labels:
                    if gender == 0:
                        label_dic_male[t].append(i)
                    else:
                        label_dic_female[t].append(i)

        # indices = [122, 128, 129, 133, 134, 135, 162, 273, 274, 366, 367, 368, 369, 371, 372, 375, 402, 404, 408, 409, 410, 412, 629, 630, 634, 635, 636, 638, 683, 947, 958, 959, 960, 961, 987, 990, 993, 994, 995, 996, 997, 998, 999, 1001, 1222, 1227, 1233, 1240, 1264, 1265, 1268, 1270, 1539, 1540, 1541, 1544, 1545, 1546, 1547, 1549, 1550, 1551, 1552, 1553, 1554, 1830, 1831, 1834, 1835, 1836, 1837, 1840, 1841, 1846, 1848, 2146, 2147, 2387, 2388, 2604, 2605, 2606, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2648, 2906, 2907, 2913, 3073, 3198, 3199, 3207, 3238, 3253, 3254, 3256, 3257, 3258, 3259, 3260, 3495, 3496, 3499, 3500, 3505, 3507, 3509, 3547, 3548, 3551, 3777, 4019, 4023, 4032, 4034, 4297, 4298, 4303, 4304, 4307, 4310, 4339, 4341, 4343, 4346, 4347, 4348, 4349, 4582, 4585, 4590, 4591, 4592, 4593, 4595, 4596, 4599, 4600, 4608, 4610, 4812, 4814, 4815, 4816, 4817, 4820, 4821, 5038, 5040, 5041, 5043, 5044, 5045, 5088, 5089, 5307, 5308, 5315, 5316, 5317, 5496, 5498, 5499]
        
        for t in target_labels:
            for g in [0,1]:
                val_subset = torch.utils.data.Subset(val_dataset, label_dic_female[t])
                val_subset_loader = data.DataLoader(val_subset, batch_size=32, shuffle=True,  num_workers=0)
                num_sample = len(val_subset)
                print("number of sample: {0}".format(num_sample))
                explain_group(model, val_subset_loader, adj, t, g, args, method = args.method)
    else:
        explain(model, val_loader, adj, args, method = args.method)
    print("end explain")




