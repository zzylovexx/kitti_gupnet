import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
import operator
class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'seg_loss':[],
                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'], 
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss']}                                 
    def compute_weight(self,current_loss,epoch):
        T=140
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)),1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]      
                    loss_weights[current_topic] = time_value**(1-control_weight)
            #pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)   
        return loss_weights
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class GupnetLoss(nn.Module):
    def __init__(self,epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch
        


    def forward(self, preds, targets, task_uncertainties=None):

        seg_loss = self.compute_segmentation_loss(preds, targets)
        bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
        bbox3d_loss = self.compute_bbox3d_loss(preds, targets)

        heading_loss,grouploss = compute_heading_loss(preds['heading'], targets['indices'],targets['mask_2d'],targets['heading_bin'],targets['heading_res'],targets['theta_ray'],targets['group'])

        loss = seg_loss + bbox2d_loss + bbox3d_loss+heading_loss + 2*grouploss
        self.stat['heading_loss']=heading_loss
        self.stat['group_loss']=grouploss
       
        
        # self.stat['total_loss']=loss
        return loss, self.stat


    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss


    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        
        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')
         

        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        
        # compute depth loss        
        depth_input = input['depth'][input['train_tag']] 
        depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
        #depth_input, depth_log_variance = depth_input[:, 0], depth_input[:, 1]

        depth_target = extract_target_from_tensor(target['depth'], target[mask_type])
        depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)
        
        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]  
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
        
        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']] 
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
               laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])/3
        #size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')+\
        #       laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])
        # compute heading loss
        '''
        heading_loss,grouploss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                            target[mask_type],
                                            target['group'],
                                            target['theta_ray'],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])
        '''
        # loss = depth_loss + offset3d_loss + size3d_loss + heading_loss + grouploss
        loss = depth_loss + offset3d_loss + size3d_loss 
        
        
        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        # self.stat['heading_loss'] = heading_loss
        # self.stat['group_loss'] = grouploss
        
        
        return loss




### ======================  auxiliary functions  =======================
def myclass2angle(bin_class,residual): #group loss add 
    angle_per_class=2*torch.pi/float(12)
    #angle_per_class=0.5235987755
    angle=angle_per_class*bin_class.float()
    angle=angle+residual
    # print(angle)
    return angle

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]

#compute heading loss two stage style  
'''
def compute_heading_loss(input, mask, group,theta_ray,target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)
    group=group.view(-1)#group loss add
    theta_ray=theta_ray.view(-1)  #group loss add
    theta_ray=theta_ray[mask] #group loss add
    group_mask=group[mask]#group loss add 
    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    input_cls_max=torch.argmax(input_cls,dim=1)
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    gt_alpha=myclass2angle(target_cls,target_reg)
    gt_ry=gt_alpha +theta_ray
    regress_alpha=myclass2angle(input_cls_max,input_reg) #group loss add 
    regress_ry=theta_ray+regress_alpha
    grouploss=group_loss(regress_ry,gt_ry,group_mask)
    return cls_loss + reg_loss , grouploss
'''
def compute_group_loss(input_angle,target_angle,targert_group):
    group_target=targert_group.detach()
    unique_target=torch.unique(group_target)#targert_group中不重複的數字並組成一個新的tensor list並遍歷
    grouploss=0
    for element_group_number in unique_target:
        index=torch.where(group_target==element_group_number)[0]
        # index=index.squeeze()
        if len(index)==1:
            continue
        value_tensor_list=torch.index_select(input_angle,dim=0,index=index)
        # gt_tensor_list=torch.index_select(target_angle,dim=0,index=index)
        # value_tensor_list=torch.abs(value_tensor_list-gt_tensor_list)
        value_tensor_list=value_tensor_list.float()
        dev=torch.std(value_tensor_list) #caculate each groups stanrd varience
        grouploss+=(dev*len(value_tensor_list))#group loss在越多的集體weight 越大
    grouploss/=len(group_target)
    return grouploss

def compute_heading_loss(input, ind, mask, target_cls, target_reg,targert_theta_ray,targert_group):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    group=targert_group.view(-1)
    group=group[mask]
    theta_ray=targert_theta_ray.view(-1)
    theta_ray=theta_ray[mask]

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    

    '''
    for group loss preprocess
    '''
    regress_alpha=myclass2angle(torch.argmax(input_cls,dim=1), input_reg)
    gt_alpha=myclass2angle(target_cls,target_reg)
    regress_ry=regress_alpha+theta_ray
    gt_ry=gt_alpha+theta_ray

    grouploss=compute_group_loss(regress_ry, gt_ry, group)
    return cls_loss+reg_loss,grouploss



if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

