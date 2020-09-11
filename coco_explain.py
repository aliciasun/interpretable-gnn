#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
import heapq
import shutil
import argparse
import time
import json
from tqdm import tqdm


from networkx.readwrite import json_graph
import matplotlib as mpl



import torch

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable



import dataset
import utils
import utils_viz
from models import gcn_resnet101, Warp, gen_A
from explain import Explainer
from interpreter import GNNInterpreter



label_dic ={"airplane": 0, 
            "apple": 1, 
            "backpack": 2, 
            "banana": 3, "baseball bat": 4, "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9, "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14, "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19, "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24, "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29, "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34, "hair drier": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39, "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44, "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49, "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54, "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59, "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64, "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69, "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74, "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79}

idx2label = dict((v,k) for k,v in label_dic.items())


def explain(model, val_loader, orig_A, args, method = 'grad'):
    # male_grad = np.zeros((80, 80))
    # num_male = 0
    # female_grad = np.zeros((80, 80))
    # num_female = 0
    for i, (inp, target) in enumerate(val_loader):
        if i > 5:
            break
        print("compute saliency maps start...")
        # A=Variable(A_, requires_grad=True)
        photo=Variable(inp[0], requires_grad=True).float()
        img_path = inp[1][0].split(".")[0]
        gender = Variable(inp[2], requires_grad=False).detach().numpy()
        feature=Variable(inp[3], requires_grad=True).float()

        pred=model(photo, feature, mode='explain')
        pred_list = pred[0].detach().numpy()
        preds = list(np.where(pred_list>0.5)[0])
        pred_prob={i:pred_list[i] for i in preds}

        predicted_labels = [idx2label[l] for l in preds]
        print("predict label.....",predicted_labels)

        true_labels = [idx2label[l.numpy()[0]] for l in inp[4]]
        print("ground truth label....",true_labels)

        if method == 'grad':
   
            pred.backward(gradient=torch.ones_like(pred),retain_graph=True)
            grad_A=model.A.grad.detach()
            grad_A = grad_A.numpy()
            pos_grad = grad_A*(grad_A>=0)
            neg_grad = grad_A*(grad_A<0)

            to_keep = pos_grad*orig_A
            to_add = pos_grad*(1.0-orig_A)
            file_name = 'subgraph_coco_gender/image/{}.json'.format(img_path)
            utils_viz.save_adj_to_json(file_name, pred_prob, to_keep, to_add)
        elif method == 'mask':
            adj=torch.Tensor(orig_A)
            # set up interpreter
            interpreter = GNNInterpreter(model, adj, photo, feature, target, pred, args)

            for step in tqdm(range(args.num_steps)):
                interpreter.optimizer.zero_grad()
                loss = interpreter.loss()
                loss.backward(retain_graph=True)
                interpreter.optimizer.step()
                if step%10 == 0:
                    print('[iter:%d] mask weight: %.3f loss: %.3f' %
                    (step, torch.sum(interpreter.mask), loss))
            mask_to_add = interpreter.mask_to_add.detach().numpy()
            mask_to_keep = interpreter.mask_to_keep.detach().numpy()
            file_name = 'mask_interpreter/lambda_3/max_pred/{}.json'.format(img_path)
            utils_viz.save_graph_to_json(file_name, predicted_labels, mask_to_keep, mask_to_add)
            print(*utils.get_top_k_pairs(mask_to_keep, idx2label), sep = "\n")
            print('-------------------------------------------')
            print(*utils.get_top_k_pairs(mask_to_add, idx2label),sep = "\n")

        #save results
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.imshow(mask_to_keep)
            ax2 = fig.add_subplot(122)
            ax2.imshow(mask_to_add)
            # plt.show()
        
      






if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--steps", dest="num_steps", type=int, help="Number of iters to train the mask."
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
        num_epochs=100,
        num_steps=10
    )
    args = parser.parse_args()

    adj=gen_A(num_classes=80,t=0.4, adj_file='data/coco/coco_adj.pkl')
    adj[np.nonzero(adj)]=1  

    #use pretrained ML_GCNmodel
            
    model = gcn_resnet101(num_classes=80, t=0.4, adj_file = 'data/coco/coco_adj.pkl', interpret_mode = True)
    checkpoint=torch.load('coco_checkpoint.pth.tar',map_location=torch.device('cpu'))

                        
    model.load_state_dict(checkpoint['state_dict'],False)                     
    model.eval()
        
    val_dataset = dataset.CocoGender(phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                                std=model.image_normalization_std)
    val_dataset.transform = transforms.Compose([
                Warp(448),
                transforms.ToTensor(),
                normalize,
            ])
        
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    file_name = 'subgraph_coco_gender/ground_truth.json'
    # utils_viz.save_node_adj_to_json(file_name, adj)
    explain(model,val_loader, adj, args, method = 'mask')
    print("end explain")




