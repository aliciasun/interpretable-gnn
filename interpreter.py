import math
import torch
import torch.nn as nn
import torch.optim as optim


class GNNInterpreter:
    def __init__(self, model, adj, feature, x, labels, pred, args):
        self.model = model
        self.model.eval()
        self.num_nodes = adj.shape[0]
        self.adj = adj.clone().detach()-torch.eye(self.num_nodes)
        self.feature = feature
        self.x = x
        self.labels = labels
        self.pred = pred
        self.args = args
        self.num_nodes = 80
        adj_supplement = torch.ones_like(self.adj)-torch.eye(self.num_nodes)-self.adj
        #a candidate to add is 1, a candidate to remove is -1
        self.budget = adj_supplement-self.adj 
        
        self.mask, self.mask_bias = self._initialize_mask(init_strategy='normal', const=0.5)
        self.mask_to_add = self.mask*(self.budget>0)
        self.mask_to_keep = self.mask*(self.budget<0)
        
        self.confidence = 0.2
        self.l_existing = 0.005
        self.l_new = 0.005
        self.optimizer = optim.SGD([self.mask], lr=0.1, momentum=0.95, weight_decay=0.0)

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
        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
            nn.init.constant_(mask_bias, const)
        else:
            mask_bias = None
        return mask, mask_bias
        

    def forward(self):
        # adj = self.adj.cuda() if self.args.gpu else self.adj
        # self.mask = torch.clamp(self.mask, 0.0, 1.0)
        masked_adj = self.adj+ self.mask*self.budget
        pred = self.model(self.feature, self.x, adj = masked_adj, mode='explain')
        return pred

    def loss(self, mode = 'max_prediction'):
        scale_factor = 1
        pred = self.forward()
        real = (self.labels * pred).sum(1)
        other = ((1.-self.labels)*pred).max(1)[0]
        mask_clipped = torch.clamp(self.mask, 0, 1)
        self.mask_to_add = mask_clipped*(self.budget>0)
        self.mask_to_keep = mask_clipped*(self.budget<0)
        if mode == 'max_prediction':
            loss_factual = torch.clamp(real - other, max = self.confidence)
            loss = scale_factor*torch.sum(loss_factual) + self.l_existing*torch.norm(self.mask_to_keep, p=1) -\
                self.l_new*torch.norm(self.mask_to_add, p=1)
            loss = -1.0*loss
        return loss
