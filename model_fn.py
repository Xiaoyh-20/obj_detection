"""
File: train_fn.py
Author: xiaoyihong
Created on: 2023/12/24
Description:
    模型搭建及微调函数
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


def model_finetune(num_classes,
                   box_score_thresh=0.7,
                   box_nms_thresh=0.4, #0.5:检测框重叠度仍较大；0.4：极大值抑制效果较好
                   box_fg_iou_thresh=0.8,
                   box_bg_iou_thresh=0.3):  # debug:类别数目应该包含背景类(PASCAL VOC:1+20)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                               box_score_thresh=box_score_thresh,
                                                               box_nms_thresh=box_nms_thresh,
                                                               box_fg_iou_thresh=box_fg_iou_thresh,
                                                               box_bg_iou_thresh=box_bg_iou_thresh)
    #finetune:替换检测框的预测头，以适应自定义数据集的分类数目
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
