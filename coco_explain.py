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
import os

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
    args.mode = 'group'
    adj=torch.Tensor(orig_A)
    interpreter = GNNInterpreter(model, adj, None, None, None, None, args)
    for step in tqdm(range(args.num_steps)):
        for i, (inp, target) in enumerate(val_loader):
            interpreter.zero_grad()
            interpreter.optimizer.zero_grad()
            if interpreter.adj.grad is not None:
                interpreter.adj.grad.zero_()
                interpreter.mask.grad.zero_()
            photo=Variable(inp[0], requires_grad=True).float().to(device)
            feature=Variable(inp[3], requires_grad=True).float().to(device)
            with torch.no_grad():
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
            del photo
            del feature
            torch.cuda.empty_cache()
        if step%10 == 0:
            print('[iter:%d] to_remove mask weight: %.3f to_add mask weight: %.3f loss: %.3f' %
            (step, torch.norm(mask_existing, p=1), torch.norm(mask_to_add, p=1),torch.sum(loss)))
    mask_to_add = mask_to_add.detach().numpy()*(1-np.eye(orig_A.shape[0]))
    mask_to_keep = (orig_A-mask_existing.detach().numpy())*(1-np.eye(orig_A.shape[0]))
    
    print(utils.get_top_k_pairs(mask_to_keep, idx2label, k=10), sep = "\n")
    print('-------------------------------------------')
    if not os.path.exists('gender/'):
        os.makedirs('gender/')
    with open('gender/pred_{0}_{1}.npz'.format(target_label,subgroup), 'wb') as f:
        np.savez(f, to_keep = mask_to_keep, to_add = mask_to_add)
    f.close()


def explain(model, val_loader, orig_A, args, method = 'mask'):
    hit_pred, hit_real = [], []
    perf_pred, perf_real = [], []
    perf_pred_add, perf_real_add = [], []
    common_pred, common_real = [], []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    for i, (inp, target) in enumerate(val_loader):
        if i<4500:
           continue
        print("training for image: {0}".format(i))
        photo=Variable(inp[0], requires_grad=True).float().to(device)
        img_path = inp[1][0].split(".")[0]
           
        if args.dataset == 'coco':
            feature=Variable(inp[2], requires_grad=True).float().to(device)
        else:
            feature=Variable(inp[3], requires_grad=True).float().to(device)
        label_idx = np.where(target.cpu()==1)[1]
        true_labels = [idx2label[l] for l in label_idx]
        
        true_label_length = len(true_labels)
        print("ground truth label....",true_labels)
        with torch.no_grad():
            pred=model(photo.to(device), feature.to(device))
            orig_pred_prob = torch.sigmoid(pred)
        pred_list = torch.sigmoid(pred[0]).cpu().detach().numpy()
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
                    interpreter.mask.grad.data[interpreter.budget>0]= 0
                elif args.mode == 'promote_v2':
                    interpreter.mask.grad.data[interpreter.budget<0]= 0
                interpreter.mask.grad.data.fill_diagonal_(0)
                # causal_edges = torch.transpose(interpreter.budget<0, 0,1)
                # interpreter.mask.grad[causal_edges]=0
                interpreter.optimizer.step()                
                interpreter.exp_lr_scheduler.step()
                mask = torch.sigmoid(interpreter.mask.detach())
                # if args.mode == 'preserve' or args.mode == 'promote_v2':
                #     mask_to_add = interpreter.mask_to_add.detach()
                # else:
                mask_existing = mask*(interpreter.budget<0)
                mask_to_add = mask*(interpreter.budget>0)
                if step%10 == 0:
                    print('[iter:%d] to_remove mask weight: %.3f to_add mask weight: %.3f loss: %.3f' %
                    (step, torch.norm(mask_existing, p=1), torch.norm(mask_to_add, p=1),torch.sum(loss)))
            mask_to_add = mask_to_add.cpu().numpy()*(1-np.eye(orig_A.shape[0]))
            mask_to_keep = (orig_A-mask_existing.cpu().numpy())*(1-np.eye(orig_A.shape[0]))
            if args.save_mask:
                with open('mask_interpreter/mask/{}/{}.npz'.format(args.mode, img_path), 'wb') as f:
                    np.savez(f, to_keep = mask_to_keep, to_add =mask_to_add)
            if args.mode == 'preserve':
                mask_to_add = np.zeros(orig_A.shape[0])
            if args.mode == 'promote_v2':
                max_index=utils.largest_indices(mask_to_add,3)
                mask_to_add[max_index] = 1
                mask_to_add[mask_to_add<1] = 0
                mask_to_keep = orig_A
            if args.mode == 'preserve' or args.mode == 'promote_v2':
                masked_adj_smooth = torch.Tensor(smooth_adj(interpreter.masked_adj.detach().cpu().numpy())).to(device)

            else:
                masked_adj_smooth = interpreter.masked_adj.detach()
            new_pred=model(photo, feature, adj = masked_adj_smooth)
            new_pred = torch.sigmoid(new_pred)
            new_pred_list = new_pred[0].cpu().detach().numpy()
            new_preds = list(new_pred_list.argsort()[-true_label_length:][::-1])
            del photo
            del feature
            torch.cuda.empty_cache()
            new_pred_prob={i:new_pred_list[i] for i in new_preds}
            new_predicted_labels = [idx2label[l] for l in new_preds]
            print("predict label.....",new_predicted_labels)
            
            common_object = len(list(set(preds).intersection(new_preds)))
            common_pred.append(common_object/len(preds))
            common_object_real = len(list(set(label_idx).intersection(new_preds)))
            common_real.append(common_object_real/len(true_labels))
           
            if args.print:
                print(*utils.get_top_k_pairs(mask_to_keep, idx2label, k=10), sep = "\n")
                print('-------------------------------------------')
                print(*utils.get_top_k_pairs(mask_to_add, idx2label, k=10), sep = "\n")
           
                # file_name = 'mask_interpreter/json/{}/{}.json'.format(args.mode,img_path)
                # utils_viz.save_adj_to_json(file_name, pred_prob, new_pred_prob, mask_to_keep, mask_to_add)

        print("common pred: {0}".format(statistics.mean(common_pred)))
        print("common real: {0}".format(statistics.mean(common_real)))

        for k in [1,2]:
            print("evaluating on {0}-hop neighbors".format(k))
            num_rel_real, num_rel_pred, num_hit_real, num_hit_pred,num_path = evaluate(label_idx, preds, orig_A, mask_to_keep, args,k=k)
            if args.mode == 'promote_v2':
                num_rel_real_add, num_rel_pred_add, _, _,num_path_add = evaluate(label_idx, preds, orig_A, mask_to_add+mask_to_keep,args, k=k)
            if num_rel_pred != 0 :
                hit_pred.append(num_hit_pred)
                perf_pred.append(num_hit_pred/num_rel_pred)
            if num_rel_real != 0:
                hit_real.append(num_hit_real)
                perf_real.append(num_hit_real/num_rel_real)

            print("number of relevent predict edges: {0}".format(num_rel_pred))
            print("number of relevent real edges: {0}".format(num_rel_real))
            print("number of hit predict edges: {0}".format(num_hit_pred))
            print("number of hit real edges: {0}".format(num_hit_real))
            print("average performance pred: {0}".format(statistics.mean(perf_pred)))
            print("average performance real: {0}".format(statistics.mean(perf_real)))
            if args.mode == 'promote_v2':
                print("number of relevent real edges after add: {0}".format(num_rel_real_add))
                print("number of paths between real labels: {0}".format(num_path))
                print("number of paths between real labels after add: {0}".format(num_path_add))
           
            # if args.mode != 'preserve':
            #     _, num_rel_add, num_hit_real_add, num_hit_pred_add,_ = evaluate(label_idx, preds, orig_A, mask_to_keep+mask_to_add, args, k=k)
            #     if num_rel_pred != 0 :
            #         perf_pred_add.append(num_hit_pred_add/num_rel_pred)
            #     if num_rel_real != 0:
            #         perf_real_add.append(num_hit_real_add/num_rel_real)
            #     print("average performance real after add: {0}".format(statistics.mean(perf_real_add)))
            #     print("average performance pred after add: {0}".format(statistics.mean(perf_pred_add)))
            


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
        "--print", 
        dest="print", 
        action="store_const", 
        const=True,
        default=False, 
        help="print"
    )
    parser.add_argument(
        "--save_mask", 
        dest="save_mask", 
        action="store_const", 
        const=True,
        default=False, 
        help="whether to save mask"
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

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("cuda: {0}".format(use_cuda))
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.multiprocessing.set_start_method('spawn')
    adj=gen_A(num_classes=80,t=0.4, adj_file='data/coco/coco_adj.pkl', p=1)
    #use pretrained ML_GCNmodel
    model = gcn_resnet101(num_classes=80, t=0.4, adj_file = 'data/coco/coco_adj.pkl', interpret_mode = True)
    checkpoint=torch.load('coco_checkpoint.pth.tar',map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'],False)                     
    model.eval()

    if args.dataset == 'coco':
        val_dataset = COCO2014(root='data/coco', phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    else:
        # train_dataset = CocoGender(phase='train', inp_name='data/coco/coco_glove_word2vec.pkl')
        val_dataset = CocoGender(phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                                std=model.image_normalization_std)
    val_dataset.transform = transforms.Compose([
                Warp(448),
                transforms.ToTensor(),
                normalize,
            ])
    # train_dataset.transform = transforms.Compose([
    #             Warp(448),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    # file_name = 'subgraph_coco_gender/ground_truth.json'
    # utils_viz.save_node_adj_to_json(file_name, adj)
    if args.mode == 'group':
        target_labels =  ['tennis racket','knife','motorcycle']
        from collections import defaultdict
        label_dic_male = defaultdict(list)
        label_dic_female = defaultdict(list)
        for i, (inp, target) in enumerate(val_loader):
            gender = Variable(inp[2], requires_grad=False).detach().numpy()[0]
            label_idx = np.where(target==1)[1]
            true_labels = [idx2label[l] for l in label_idx]
            for t in target_labels:
                if t in true_labels:
                    if gender == 1:
                        label_dic_male[t].append(i)
                    else:
                        label_dic_female[t].append(i)

        for t in target_labels:
            for g in [0,1]:
                val_subset = torch.utils.data.Subset(val_dataset, label_dic_female[t])
                val_subset_loader = data.DataLoader(val_subset, batch_size=32, shuffle=True,  num_workers=0)
                num_sample = len(val_subset)
                print("number of sample: {0}".format(num_sample))
                explain_group(model, val_subset_loader, adj, t, g, args, method = args.method)
        # val_subset = torch.utils.data.Subset(val_dataset, [1,2,3,4,5])
        # val_subset_loader = data.DataLoader(val_subset, batch_size=5, shuffle=True,  num_workers=0)
        explain_group(model, val_subset_loader, adj, 'person', 0, args, method = args.method)
    else:
        explain(model, val_loader, adj, args, method = args.method)
    print("end explain")




