import json
import os
import pickle
import pandas as pd
import numpy as np
import subprocess
from PIL import Image
from urllib.request import urlretrieve

import torch.utils.data as data

import utils

urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

id2obj = {0: 'toilet', 1: 'teddy_bear', 2: 'sports_ball', 3: 'bicycle', 4: 'kite', 5: 'skis', 6: 'tennis_racket', 7: 'donut', 8: 'snowboard', 9: 'sandwich', 10: 'motorcycle', 11: 'oven', 12: 'keyboard', 13: 'scissors', 14: 'chair', 15: 'couch', 16: 'mouse', 17: 'clock', 18: 'boat', 19: 'apple', 20: 'sheep', 21: 'horse', 22: 'giraffe', 23: 'person', 24: 'tv', 25: 'stop_sign', 26: 'toaster', 27: 'bowl', 28: 'microwave', 29: 'bench', 30: 'fire_hydrant', 31: 'book', 32: 'elephant', 33: 'orange', 34: 'tie', 35: 'banana', 36: 'knife', 37: 'pizza', 38: 'fork', 39: 'hair_drier', 40: 'frisbee', 41: 'umbrella', 42: 'bottle', 43: 'bus', 44: 'zebra', 45: 'bear', 46: 'vase', 47: 'toothbrush', 48: 'spoon', 49: 'train', 50: 'airplane', 51: 'potted_plant', 52: 'handbag', 53: 'cell_phone', 54: 'traffic_light', 55: 'bird', 56: 'broccoli', 57: 'refrigerator', 58: 'laptop', 59: 'remote', 60: 'surfboard', 61: 'cow', 62: 'dining_table', 63: 'hot_dog', 64: 'car', 65: 'cup', 66: 'skateboard', 67: 'dog', 68: 'bed', 69: 'cat', 70: 'baseball_glove', 71: 'carrot', 72: 'truck', 73: 'parking_meter', 74: 'suitcase', 75: 'cake', 76: 'wine_glass', 77: 'baseball_bat', 78: 'backpack', 79: 'sink'}
label_dic = {"airplane": 0, "apple": 1, "backpack": 2, "banana": 3, "baseball_bat": 4, "baseball_glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9, "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14, "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19, "cat": 20, "cell_phone": 21, "chair": 22, "clock": 23, "couch": 24, "cow": 25, "cup": 26, "dining_table": 27, "dog": 28, "donut": 29, "elephant": 30, "fire_hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34, "hair_drier": 35, "handbag": 36, "horse": 37, "hot_dog": 38, "keyboard": 39, "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44, "mouse": 45, "orange": 46, "oven": 47, "parking_meter": 48, "person": 49, "pizza": 50, "potted_plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54, "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59, "snowboard": 60, "spoon": 61, "sports_ball": 62, "stop_sign": 63, "suitcase": 64, "surfboard": 65, "teddy_bear": 66, "tennis_racket": 67, "tie": 68, "toaster": 69, "toilet": 70, "toothbrush": 71, "traffic_light": 72, "train": 73, "truck": 74, "tv": 75, "umbrella": 76, "vase": 77, "wine_glass": 78, "zebra": 79}

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx

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
        target = np.zeros(self.num_classes, np.float32) #- 1
        target[labels] = 1
        return (img, filename, self.inp,labels), target


#coco gender specific data
class CocoGender(data.Dataset):
    def __init__(self, transform=None, phase='train', inp_name=None):
        self.phase = phase
        self.data = pickle.load(open('data/coco_gender/{0}.data'.format(phase),'rb'))
        self.img_list = []
        self.occurrence_threshold = 100
        download_coco2014('data/coco', phase)
        print(len(self.data))
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        with open('data/coco/coco_adj.pkl', 'rb') as f:
            self.adj = pickle.load(f)
        self.inp_name = inp_name
        # self.target = np.array([sample['annotation'] for sample in self.test_data])

    def get_anno(self):
        list_path = os.path.join('data/coco_gender', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join('data/coco', 'data', 'category.json'), 'r'))

        # gender_verbs_raw = pd.read_csv(open('data/coco_gender/objs','r'), header=None)
        # gender_verbs = [label_dic[verb] for verb in list(gender_verbs_raw[0].values)]
        # self.cat2idx = gender_verbs
        # print(self.cat2idx)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        gender = item['gender']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join('data/coco/', 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) #- 1
        target[labels] = 1
        return (img, filename, gender, self.inp), target

    def get_constraints(self, margin, val = 0.0):
        margin2 = margin
        count_train = self.compute_male_female_per_object_322(self.train_data)
        train_ratio = {}
        constraints = {}
        obj_cons_train = []
        for k in count_train:
            if (count_train[k][0] + count_train[k][1]) > self.occurrence_threshold:
                obj_cons_train.append(k)
                train_ratio[k] = float(count_train[k][0]) / (count_train[k][0] + count_train[k][1])
        for k in train_ratio:
            constraints[k] = (((train_ratio[k]- 1 - margin), (train_ratio[k] - margin)), # tr-margin<= m/(m+f)
                                ((1 - margin2 - train_ratio[k]), -(margin2 + train_ratio[k])),
                                        val)
        return constraints, obj_cons_train

    def get_partial_cons(self, all_cons, cons_verbs):
        
        partial_cons = {verb:all_cons[verb] for verb in cons_verbs if verb in all_cons}
        return partial_cons

    def compute_male_female_per_object_322(self, samples):
        """
        for each label, compute occurance of male and female in training set
        """
        count = dict()
        for i in range(80):
            count[i] = [0,0]
        for sample in samples:
            sample = sample['annotation']
            if sample[0] == 1: #man
                objs = sample[2:162]
                for j in range(80):
                    if objs[2*j] == 1:
                        count[j][0] += 1
            else:#woman
                objs = sample[162:]
                for j in range(80):
                    if objs[2*j] == 1:
                        count[j][1] += 1
        return count

    def generate_adj_file(self, samples):
        #p(L_i|L_j)
        M = np.zeros((self.num_classes+2, self.num_classes+2))
        for sample in samples:
            pass

    def generate_id_gender_pair(self):
        #save id-gender-label-pair to json
        output_path = os.path.join('data/coco_gender', '{}_anno.json'.format(self.phase))
        val_anno = json.load(open('data/coco/data/val_anno.json'))
        img_list = {a['file_name']:a['labels'] for a in val_anno}
        anno_data_list = []
        for sample in self.data:
            gender = sample['annotation'][0]
            anno_data = {}
            anno_data['file_name'] = sample['img']
            anno_data['gender'] = gender
            anno_data['labels'] = img_list[sample['img']]
            anno_data_list.append(anno_data)
        f = open(output_path, 'w')
        f.write(json.dumps(anno_data_list))
        f.close()


val_dataset = CocoGender(phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
val_dataset.generate_id_gender_pair()

