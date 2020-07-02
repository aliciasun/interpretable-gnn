#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:14:16 2020

@author: lizan
"""

import argparse

from sklearn.model_selection import train_test_split
import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter
import torch.utils.data as tordata
import scipy 
from math import sin,cos,exp
from sympy import diff,symbols

from torch.autograd import Variable
from captum.attr import IntegratedGradients
urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def download_coco2014(root, phase):
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    data = os.path.join(root, 'data/')
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file,data)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.Popen('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target
    
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

A_=gen_A(num_classes=80,t=0.4, adj_file='data/coco/coco_adj.pkl')
A_=torch.from_numpy(A_)
#A=gen_A(num_classes=80,t=0.4, adj_file='/Users/lizan/Desktop/data/coco/coco_adj.pkl')
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        
        
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp, A):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)


        inp = inp[0]
        
        adj = gen_adj(Parameter(A.float()))
        
        
        x = self.gc1(inp, adj)
        
        x = self.relu(x)
        x = self.gc2(x, adj)
        #print(x)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        
        return x



   

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



def gcn_resnet101(num_classes, t, pretrained=True, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, in_channel=in_channel)

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)
#use pretrained ML_GCNmodel
        
model = gcn_resnet101(num_classes=80, t=0.4)
checkpoint=torch.load('coco_checkpoint.pth.tar',map_location=torch.device('cpu'))

                     
model.load_state_dict(checkpoint['state_dict'],False)                     
model.eval()






 


   
    
def get_val_result():
    
    global pred_list
    global val_loader
    pred_list1 =[]

    pred_list2=[]

    use_gpu = torch.cuda.is_available()

    print('val starting')
    val_dataset = COCO2014('data/coco', phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
    val_dataset.transform = transforms.Compose([
                Warp(448),
                transforms.ToTensor(),
                normalize,
            ])
    
    #use_dataset,unuse_dataset = train_test_split(val_dataset, test_size=0.01)
    val_loader = tordata.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    for i, (input, target) in enumerate(val_loader):
        
        
        
        if i > 0:
            break
        
        
        #pred1=model(input[0],input[2],torch.mul(A_,torch.FloatTensor(80, 80).uniform_(1, 1)))
        pred1=model(input[0],input[2],torch.eye(80))
        pred2=model(input[0],input[2],A_)
        #print("get the result'{}".format(pred1-pred2))
        pred_list1.append(pred1.data.numpy())
        pred_list2.append(pred2.data.numpy())
        #print("get the result'{}".format(pred1-pred2))
    pred_list1 = np.concatenate(pred_list1)
    
    pred_list2 = np.concatenate(pred_list2)
    
    #use_gpu = torch.cuda.is_available()
    
    print("get the final result'{}".format(pred_list1-pred_list2))
    


    

get_val_result()
   





model.eval()

for i, (input, target) in enumerate(val_loader):
    print("compute saliency maps start...")
    A=Variable(A_, requires_grad=True)
    
    photo=Variable(input[0], requires_grad=True)
    feature=Variable(input[2], requires_grad=True)
    pred=model(photo,feature,A)
    

    pred.backward(gradient=torch.ones_like(pred),retain_graph=True)
    print(A.grad)
    print(feature.grad)
    print(photo.grad)
    #saliency=compute_saliency_maps(input[0],input[2], A_, model)
   #print(saliency)
    if i >2:
        break 


