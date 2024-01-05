"""
File: data_process_fn.py
Author: xiaoyihong
Created on: 2023/11/30
Description:
    输入数据处理函数
    数据增强函数
    输出数据处理函数
"""
import os
import random
import xml.etree.ElementTree as ET

import numpy
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# PASCAL VOC数据集数据类别
from torchvision import transforms

num_classes = 1 + 20
classes = ["background",  # always index 0
           "aeroplane",
           "bicycle",
           "bird",
           "boat",
           "bottle",
           "bus",
           "car",
           "cat",
           "chair",
           "cow",
           "diningtable",
           "dog",
           "horse",
           "motorbike",
           "person",
           "pottedplant",
           "sheep",
           "sofa",
           "train",
           "tvmonitor"]

"""
xml_parser(file_path):解析xml标注文件，返回包含目标类别和标注框相对值的标注数据
"""
def xml_parser(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    #   输出targets格式规范：
    # - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
    #   ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
    # - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    # - masks (``UInt8Tensor[N, H, W]``): the segmentation binary masks for each instance
    targets = {"labels": [], "boxes": [], "masks": []}
    for idx, obj in enumerate(root.iter('object')):
        cls_name = obj.find("name").text
        cls_inx = torch.tensor(classes.index(cls_name))

        bndbox = obj.find("bndbox")
        bbox = torch.tensor([float(bndbox.find("xmin").text),
                             float(bndbox.find("ymin").text),
                             float(bndbox.find("xmax").text),
                             float(bndbox.find("ymax").text)])
        targets["labels"].append(cls_inx)
        targets["boxes"].append(bbox)

        #尝试使用检测框信息生成粗略的 mask
        mask_img = torch.zeros((1, h, w))  # debug：Resize需要输入至少三维张量，
        bbox_int = torch.round(bbox).int()  # debug：索引必须是整数 #引入了一定的量化误差
        mask_img[0, bbox_int[1] - 1:bbox_int[3], bbox_int[0] - 1:bbox_int[2]] = torch.tensor(
            1)  # debug：binary masks for each instance
        targets["masks"].append(mask_img)

    targets["labels"] = torch.tensor(torch.stack(targets["labels"]), dtype=torch.int64)
    targets["boxes"] = torch.tensor(torch.stack(targets["boxes"]), dtype=torch.float)
    targets["masks"] = torch.tensor(torch.stack(targets["masks"]), dtype=torch.uint8)
    return targets


"""
MyDataset(root_dir, images_dir, labels_dir, image_transform = None, label_transform = None):自定义数据集
"""
class MyDataset(Dataset):
    def __init__(self, root_dir, images_dir, labels_dir, image_transform=None, label_transform=None):
        self.root_dir = root_dir
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.images_path = os.path.join(self.root_dir, self.images_dir)
        self.labels_path = os.path.join(self.root_dir, self.labels_dir)

        self.images_list = [file for file in os.listdir(self.images_path) if
                            file.endswith('jpg')]  # debug# ：读取了macOS系统自定义属性文件
        self.labels_list = os.listdir(self.labels_path)
        self.images_list.sort()
        self.labels_list.sort()
        if image_transform is not None:
            self.image_transform = image_transform
        if label_transform is not None:
            self.label_transform = label_transform

    def __getitem__(self, inx):
        image_name = self.images_list[inx]
        label_name = self.labels_list[inx]
        image_path = os.path.join(self.root_dir, self.images_dir, image_name)
        label_path = os.path.join(self.root_dir, self.labels_dir, label_name)

        image = Image.open(image_path)
        label = self.label_transform(label_path)
        #debug:需要同时输入image和label，保证数据增强后的数据为有效数据
        if self.image_transform:
            image = self.image_transform(image)

        return image, label

    def __len__(self):
        assert len(self.images_list) == len(self.labels_list)
        return len(self.images_list)


"""
collate_fn(batch):实现将batch中维度不一致的数据进行拼接，返回拼接后的数据
    :batch: 输入dataloader加载的batch，其中每个元素为一个tuple，包含image和target
    :return: 输入imgaes列表和对应targets列表,其中每个元素为一个dict，包含labels,boxes,masks
"""
def my_collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets

"""
MyTransform:自定义数据增强类,使用了模糊、颜色抖动
"""
class MyTransform:
    def __init__(self, flip_prob=0.5, jitter_prob=0.5, crop_prob=0.5, blur_prob=0.5):
        self.flip_prob = flip_prob
        self.jitter_prob = jitter_prob
        self.crop_prob = crop_prob
        self.blur_prob = blur_prob

    def __call__(self, image):
        w,h=image.size
        # 随机颜色抖动
        if random.random() < self.jitter_prob:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(image)

        # 随机模糊
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.BLUR)

        image=transforms.ToTensor()(image)
        return image

"""
def criterion_mAP():自定义验证集评价指标mAP
"""
def criterion_mAP(ground_truths, preds, num_classes, tp_thres, fp_thres):
    gt_boxes=[]
    gt_labels=[]
    pred_boxes=[]
    pred_labels=[]
    scores=[]
    
    AP_all_classes=[]
    # print("mAP calculation starts:\n")

    #debug：gt_batch是一个list，其中每个元素为一个dict，包含labels,boxes,masks
    #debug：pred_batch是一个list，其中每个元素为一个dict，包含labels,boxes,masks
    temp=[]
    for gt_batch in ground_truths:
        temp.extend(gt_batch)
    ground_truths=temp#[{}]
    temp=[]
    for pred_batch in preds:
        temp.extend(pred_batch)#[{}]
    preds=temp

    for gt in ground_truths: #gt{}:one image
        #提取每一个ground truth的boxes和labels
        if gt["boxes"].dim()==1:
            gt_boxes.append(gt["boxes"])
            gt_labels.append(gt["labels"])
        else:
            for i in range(gt["boxes"].shape[0]):
                gt_boxes.append(gt["boxes"][i]) #debug:区分一维张量和高维张量#gt_boxes[tensor]
                gt_labels.append(gt["labels"][i]) #debug:0-dim tensor

    for pred in preds:
        #提取每一个预测的boxes和labels
        if pred["boxes"].dim()==1:
            pred_boxes.append(pred["boxes"])
            pred_labels.append(pred["labels"])
        else:
            for i in range(pred["boxes"].shape[0]):
                pred_boxes.append(pred["boxes"][i])
                pred_labels.append(pred["labels"][i]) #debug:0-dim tensor)
                scores.append(pred["scores"][i]) #debug:0-dim tensor

    assert len(pred_boxes) == len(scores) == len(pred_labels)
    assert len(gt_boxes) == len(gt_labels)
    # print(f"pred_boxes:{len(pred_boxes)},scores:{len(scores)},pred_labels:{len(pred_labels)},gt_boxes:{len(gt_boxes)},gt_labels:{len(gt_labels)}")
    num_classes = num_classes
    average_precisions = torch.zeros((num_classes - 1), dtype=torch.float)  # debug:不计算背景类

    for c in range(1, num_classes): #input:[tensor]
        pb_per_class = []  # pred_boxed_per_class
        ps_per_class = []  # pred_scores_per_class
        gtb_per_class = []  # ground_truth_boxes_per_class

        for i in range(len(pred_boxes)):
            if pred_labels[i] == c:
                pb_per_class.append(pred_boxes[i])
                ps_per_class.append(scores[i])
        for i in range(len(gt_boxes)):
            if gt_labels[i] == c:
                gtb_per_class.append(gt_boxes[i])
        # print(f"c:{c}")
        # print(f"pred_boxes:{pb_per_class}\t scores:{ps_per_class}\n")
        # print(f"gt_boxes:{gtb_per_class}")


        #数据转化为张量
        if len(pb_per_class) != 0:
            if len(pb_per_class) == 1:
                pb_per_class = (pb_per_class[0]).unsqueeze(0)   # [num_pred_per_class, 4] #考虑 pb_per_class = 空列表的情况
            else:
                pb_per_class =(torch.cat(pb_per_class,dim=0)).view((-1,4)) #[num_pred_per_class, 4]
        if len(gtb_per_class) != 0:
            if len(gtb_per_class) == 1:
                gtb_per_class = (gtb_per_class[0]).unsqueeze(0)
            else:
                 gtb_per_class =(torch.cat(gtb_per_class,dim=-1)).view((-1,4)) #[num_gt_per_class, 4]


        #debug:没有真值目标时,平均精度为0
        #debug:没有检测出目标时,平均精度为0
        num_gts =len(gtb_per_class)
        num_preds = len(pb_per_class)

        if num_gts == 0 and num_preds == 0:
            continue
        elif num_gts == 0 and num_preds != 0:
            average_precisions[c - 1] = 0 #debug:物体类别从1开始计数
            continue
        elif num_gts != 0 and num_preds == 0:
            average_precisions[c - 1] = 0
            continue
        else:
            # 按照置信度对pred_boxes排序
            temp=[]
            for score in ps_per_class:
                temp.append(score.item())
            ps_per_class=torch.tensor(temp)
            sorted_descend=torch.argsort(ps_per_class, dim=0, descending=True) #debug：此处 scores 是列表，不是张量

            temp=pb_per_class
            for i in range(len(sorted_descend)):#debug：忘记了提取对应类别的置信度
                pb_per_class[i]=temp[sorted_descend[i].item()]

            # 计算 TP and FP
            tp_per_pred=torch.zeros((num_preds), dtype=torch.float)
            fp_per_pred=torch.zeros((num_preds), dtype=torch.float)

            ious = iou(pb_per_class, gtb_per_class)#计算所有预测框和真值框的iou
            print(f"ious:{ious}")
            max_iou, matched_idx = torch.max(ious, dim=0)#返回每个真值框对应的预测框iou最大值和索引
            
            for i in range(num_preds):
                pred_box=pb_per_class[i]

                #debug:同一个真值框只能对应一个预测框
                if (i in matched_idx):
                    j=0
                    for j in range(len(matched_idx)):
                        if matched_idx[j]==i:
                            gt_idx=j
                            break
                    # gt_idx=torch.where(matched_idx==i)[0]#debug:torch.where返回的是tuple
                    #box_fg_iou_thresh box_bg_iou_thresh
                    if ious[i,gt_idx] >tp_thres:
                        tp_per_pred[i] = 1
                    elif ious[i,gt_idx] < fp_thres:
                         fp_per_pred[i] = 1
            tp_per_pred=torch.tensor(tp_per_pred).view(-1,1)
            fp_per_pred=torch.tensor(fp_per_pred).view(-1,1)
        # print(f"tp_per_pred:{tp_per_pred}\nfp_per_pred:{fp_per_pred}")
        #计算累计的TP和FP
        tp_sum=torch.cumsum(tp_per_pred,dim=0) #debug:torch.cumsum返回的是二维张量 N*1
        fp_sum=torch.cumsum(fp_per_pred,dim=0)
        # print(f"tp:{tp_sum}\nfp:{fp_sum}")
        #计算recall和precision
        precision=tp_sum/(tp_sum+fp_sum)
        recall=tp_sum/num_gts
        # print(f"precision:{precision}\nrecall:{recall}")
        #计算AP
        AP_per_class=ap(recall,precision)
        AP_all_classes.append(AP_per_class)
        # print(f"AP:{AP_per_class}")
    
    #计算mAP
    if len(AP_all_classes)==0:
        mAP=torch.tensor(0,dtype=torch.float32)
        # print(f"AP_all_classes:{AP_all_classes}")
    else:
        # print(f"AP_all_classes:{AP_all_classes}")
        temp=torch.stack(AP_all_classes,dim=0)
        temp=torch.tensor(temp,dtype=torch.float32) #debug：mean input:只能是 float 或者 complex
        mAP=torch.mean(temp)#debug:需要考虑 AP均为0的情况
    return mAP

# 计算每一个检测框和所有真值框的iou          
def iou(pred_boxes, gt_boxes):#return [num_preds, num_gts]
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if gt_boxes.dim() == 1:
        gt_boxes = gt_boxes.unsqueeze(0)
    iou_matrix=torch.zeros((pred_boxes.size(0), gt_boxes.size(0)),dtype=torch.float)# preds * gts
    
    for pb_idx, pb in enumerate(pred_boxes):
        for gt_idx, gt in enumerate(gt_boxes):
            # 计算预测框和真值框的iou
            i_xmin=torch.max(pb[0], gt[0]).item()
            i_ymin=torch.max(pb[1], gt[1]).item()
            i_xmax=torch.min(pb[2], gt[2]).item()
            i_ymax=torch.min(pb[3], gt[3]).item()
            i_w=torch.max(torch.tensor(i_xmax-i_xmin+1.), torch.tensor(0)).item()
            i_h=torch.max(torch.tensor(i_ymax-i_ymin+1.), torch.tensor(0)).item()
            inter=i_w*i_h

            union=((pb[2]-pb[0]+1.)*(pb[3]-pb[1]+1.)+(gt[2]-gt[0]+1.)*(gt[3]-gt[1]+1.)).item()-inter
            iou=inter/union
            # pb_idx=torch.where(pred_boxes==pb)[0]
            # gt_idx=torch.where(gt_boxes==gt)[0]
            iou_matrix[pb_idx,gt_idx]=iou
    return iou_matrix

# 计算AP
# 此处使用11点插值法计算AP
def ap(recall, precision):
    ap=0
    recall=recall.squeeze(1) #debug：输入的 recall 和 precision 都是二维张量，但是只有一列，所以要将其压缩成一维张量
    precision=precision.squeeze(1)
    sorted_rec_idx=torch.argsort(recall, descending=True)
    recall=recall[sorted_rec_idx]
    precision=precision[sorted_rec_idx]

    max_prec=torch.zeros((11,),dtype=torch.float)

    for i in numpy.arange(0., 1.1, 0.1):
        temp=[]
        for j in range(recall.size(0)):
            if recall[j]>=i:
                temp.append(precision[j].item())
        precision=torch.tensor(temp)
        if precision.nelement() == 0:
            max_prec[int(i * 10)] = 0
        else:
            max_prec[int(i * 10)] = torch.max(
                precision)  # debug：如果 recall 中没有元素大于或等于 i，那么 precision[recall>=i] 将是一个空的张量，对空的张量调用 torch.max 将返回 nan

    ap = torch.sum(max_prec) / 11.
    return ap






