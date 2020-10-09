import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def smooth_adj(adj, p=0.25):
    adj = adj * p / (adj.sum(0, keepdims=True) + 1e-6)
    adj = adj + np.identity(adj.shape[0], np.int)
    return adj

class GNNInterpreter(nn.Module):
    def __init__(self, model, adj, feature, x, labels, pred, args
    ):
        super(GNNInterpreter, self).__init__()
        self.model = model
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.model.eval()
        self.num_nodes = adj.shape[0]
        self.adj = adj.clone().to(self.device)
        self.feature = feature
        self.x = x
        self.labels = labels
        self.pred = pred
        self.mode = args.mode
        self.args = args
        print(args.mode)
        self.num_nodes = self.adj.shape[0]
        self.creterion = nn.MultiLabelSoftMarginLoss()
        self.target_label = args.target_label 
   
        self.diagonal_mask = torch.ones_like(self.adj).to(self.device)-(torch.eye(self.num_nodes)).to(self.device)
        self.diagonal_mask = self.diagonal_mask.to(self.device)
        adj_supplement = torch.ones_like(self.adj).to(self.device)-torch.eye(self.num_nodes).to(self.device)-self.adj
        #a candidate to add is 1, a candidate to remove is -1
        self.budget = adj_supplement.to(self.device)-self.adj 
        self.budget = self.budget.to(self.device)
        torch.manual_seed(1)

        
        if args.mode == 'promote':
            mask, mask_bias = self._initialize_mask(init_strategy='const', const=0.0)
            self.mask_to_add = mask*(self.budget>0)
            self.mask_existing = mask*(self.budget<0)
            self.mask = mask
            self.l_existing = 0.0005
            self.l_new = 0.005
            self.confidence = 0.5

        elif args.mode == 'promote_v2':
            mask, mask_bias = self._initialize_mask(init_strategy='const', const=0.5)
            self.mask = torch.nn.Parameter(mask.to(self.device))
            self.mask_to_add = self.mask*(self.budget>0)
            self.mask_existing = torch.zeros_like(self.mask)
            self.l_new = 0.0005
            self.confidence = 0.5
        elif args.mode == 'preserve' or args.mode == 'group':
            mask, mask_bias = self._initialize_mask(init_strategy='const', const=0.5)
            with torch.no_grad():
                mask[self.budget>0]=0
            self.mask = torch.nn.Parameter(mask.to(self.device))
            self.mask_existing = self.mask*(self.budget<0)
            self.mask_to_add = torch.zeros_like(self.mask)
            self.l_existing = 0.001
        elif args.mode == 'attack' or args.mode == 'target_attack':
            mask, self.bias = self._initialize_mask(init_strategy='const', const=0.0)
            self.mask = torch.nn.Parameter(mask.to(self.device))
            self.mask_to_add = self.mask*(self.budget>0)
            self.mask_existing = self.mask*(self.budget<0)
            self.l_existing = 0.001
            self.l_new = 0.001
            self.confidence = 0.1
        else:
            pass
        self.mask.to(self.device)
        # self.optimizer = optim.SGD([self.mask], lr=0.001, momentum=0.95, weight_decay=1e-4)
        self.optimizer = optim.Adam([self.mask],lr=0.1)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                             step_size=20, gamma=0.1)
 

    def _initialize_mask(self,init_strategy='normal', const = 1.0):
        mask = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (self.num_nodes + self.num_nodes)
            )
            with torch.no_grad():
                mask.normal_(const, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const)
        elif init_strategy == "seperate":
            nn.init.constant_(mask, const)
            with torch.no_grad():
                mask[self.budge>0].fill_(0)
        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
            nn.init.constant_(mask_bias, const)
        else:
            mask_bias = None
        return mask, mask_bias
        

    def forward(self, feature, x):
        self.masked_adj = self.get_masked_adj().to(self.device)
        cur_pred = self.model(feature, x, adj = self.masked_adj, mode='explain')
        cur_pred_prob = torch.sigmoid(cur_pred)
        return cur_pred, cur_pred_prob

    def get_masked_adj(self):
        mask = torch.sigmoid(self.mask)
        if self.mode == 'preserve' or self.mode == 'group':
            self.mask_existing = mask*(self.budget<0)
            masked_adj = self.adj + self.mask_existing*self.budget*self.diagonal_mask
        elif self.mode == 'promote_v2':
            self.mask_new = mask*(self.budget>0)
            masked_adj = self.adj + self.mask_new*self.budget*self.diagonal_mask
        else:
            masked_adj = self.adj + mask*self.budget*self.diagonal_mask
            self.mask_existing = mask*(self.budget<0)
            self.mask_to_add = mask*(self.budget>0)
        # masked_adj = masked_adj * 0.25 / (torch.sum(masked_adj) + 1e-6)
        masked_adj = masked_adj.to(self.device)
        masked_adj_smooth = masked_adj + torch.eye(self.num_nodes)
        return masked_adj_smooth

    def loss(self, pred, orig_pred):
        scale_factor = 1.0/self.x.shape[0]
        pred_prob = torch.sigmoid(pred).to(self.device)
        true_label_length = self.labels[self.labels==1].shape[0]
        preds_labels = list(pred_prob[0].detach().cpu().numpy().argsort()[-true_label_length:][::-1])
        pred_prob_binary = pred_prob.clone().detach()
        pred_prob_binary[0][preds_labels]=1
        pred_prob_binary[pred_prob_binary!=1]=0
        if self.mode == 'promote':
            other = ((1.0-self.labels)*pred_prob).max(1)[0]
            real = pred_prob[self.labels==1]
            loss_factual = torch.clamp(real-other, max = self.confidence)
            loss = scale_factor*torch.sum(loss_factual) - self.l_existing*torch.norm(self.mask_existing, p=1) -\
            self.l_new*torch.norm(self.mask_to_add, p=1)
            loss = -1.0*loss
        elif self.mode == 'promote_v2':
            other = ((1.0-self.labels)*pred_prob).max(1)[0]
            real = pred_prob[self.labels==1]
            loss_factual = torch.clamp(real-other, max = self.confidence)
            loss = scale_factor*torch.sum(loss_factual) - self.l_new*torch.norm(self.mask_to_add, p=1)
            loss = -1.0*loss
        elif self.mode == 'preserve':
            loss_pred = self.creterion(pred, orig_pred)
            loss = loss_pred - self.l_existing*torch.norm(self.mask_existing, p=1)
        elif self.mode == 'attack':
            real = pred_prob[pred_prob_binary==1].max()
            other = ((1.0-pred_prob_binary)*pred_prob).max(1)[0]
            loss_counter = torch.clamp(other-real, max = self.confidence)
            loss = scale_factor*torch.sum(loss_counter)-self.l_existing*torch.norm(self.mask_existing, p=1) -\
            self.l_new*torch.norm(self.mask_to_add, p=1)
            loss = -1.0*loss
        elif self.mode == 'target_attack':
            real = pred_prob[0][self.target_label]
            other = ((1.0-self.labels)*pred_prob).max(1)[0]

            real = pred_prob[pred_prob_binary==1].min()
            other = ((1.0-pred_prob_binary)*pred_prob).max(1)[0]
            loss_counter = torch.clamp(real-other, min = -1.0*self.confidence)
            loss = scale_factor*torch.sum(loss_counter)+self.l_existing*torch.norm(self.mask_existing, p=1) +\
            self.l_new*torch.norm(self.mask_to_add, p=1)
        return loss
