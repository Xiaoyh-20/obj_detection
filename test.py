"""
File: test.py
Author: xiaoyihong
Created on: 2023/12/31
Description: 
    模型测试脚本
    对测试数据集(本项目中也指验证数据集)进行推理并计算评价指标mAP
"""

import torch
# import torchinfo
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# from flashtorch import utils
import wandb
from data_process_fn import MyDataset, criterion_mAP, my_collate_fn, xml_parser, MyTransform
from model_fn import model_finetune
from train_fn import train, test

# import imgaug.augmenters as iaa
#可视化平台初始化
writer = SummaryWriter("./runs/test")
wandb.init(project="maskrcnn_test")
#清楚 GPU 缓存
torch.cuda.empty_cache()
#超参数设置
batch_size = 1
#图像处理：张量化
test_image_transforms = transforms.ToTensor()

# 设置网络模块
mydevice = ("cuda:0" if torch.cuda.is_available() else "cpu")
mymodel = model_finetune(num_classes=21,
                         box_score_thresh=0.8,
                         box_nms_thresh=0.4,
                         box_fg_iou_thresh=0.8,
                         box_bg_iou_thresh=0.3)
mymodel.to(mydevice)
mymodel.load_state_dict(torch.load("models/model_best.pth"))


mycriterion = criterion_mAP


test_data = MyDataset(root_dir="./data",
                     images_dir="images/val",
                     labels_dir="labels/val",
                     image_transform=test_image_transforms,
                     label_transform=xml_parser)

test_loader = DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=my_collate_fn)

# 开始推理
wandb.watch(mymodel, log="all")


test(1, mymodel, test_loader, mycriterion, mydevice)

writer.close()
torch.cuda.empty_cache()