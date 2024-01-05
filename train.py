"""
File: train.py
Author: xiaoyihong
Created on: 2023/11/30
Description:
    模型训练脚本
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
from train_fn import train, val

# import imgaug.augmenters as iaa

writer = SummaryWriter("./runs/train")
wandb.init(project="maskrcnn_demo")
wandb.define_metric("lr",step_metric="epoch")
torch.cuda.empty_cache()
# config=wandb.config
# 设置超参数
batch_size = 8
lr = 1e-4
epochs = 50

# data augmentation
train_image_transforms = MyTransform(flip_prob=0.5, jitter_prob=0.5, crop_prob=0.5, blur_prob=0.5)
                                             # transforms.RandomHorizontalFlip(p=0.5)])
val_image_transforms = transforms.ToTensor()

# 设置网络模块
mydevice = ("cuda:0" if torch.cuda.is_available() else "cpu")
mymodel = model_finetune(num_classes=21,
                         box_score_thresh=0.8,
                         box_nms_thresh=0.4,
                         box_fg_iou_thresh=0.8,
                         box_bg_iou_thresh=0.3)
mymodel.load_state_dict(torch.load("models/model_best.pth"))

mymodel.to(mydevice)

myoptimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(myoptimizer, step_size=10, gamma=0.1)
mycriterion = criterion_mAP


train_data = MyDataset(root_dir="./data",
                       images_dir="images/train",
                       labels_dir="labels/train",
                       image_transform=train_image_transforms,
                       label_transform=xml_parser)
val_data = MyDataset(root_dir="./data",
                     images_dir="images/val",
                     labels_dir="labels/val",
                     image_transform=val_image_transforms,
                     label_transform=xml_parser)

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=False,
                          collate_fn=my_collate_fn)
val_loader = DataLoader(val_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=my_collate_fn)

# test：测试数据加载是否成功
# for images, targets in train_loader:
#     print(f"images.shape: {images.shape}")
#     plt.imshow(images[0].permute(1, 2, 0))  # 注意：PIL：(H,W,C)；torch：(C,H,W)
#     plt.show()

# test：测试image和target是否匹配
# for images, targets in train_loader:
#     print(f"images[0].shape:{images[0].shape}")
#     img = torch.tensor(images[0]*255, dtype=torch.uint8)
#     img_box = draw_bounding_boxes(img, targets[0]["boxes"],width=5)
#     print(type(img_box))
#     plt.imshow(img_box.permute(1, 2, 0))
#     plt.show()

# 开始训练
wandb.watch(mymodel, log="all")

for epoch in range(1, epochs + 1):
    wandb.log({"epoch": epoch})

    train(epoch, mymodel, train_loader, myoptimizer, mydevice)
    my_lr_scheduler.step()
    wandb.log({"lr": my_lr_scheduler.get_last_lr()[0]})

    torch.save(mymodel.state_dict(), f'models/model_v10{epoch}.pth')
    wandb.save(f'model_v10{epoch+50}.pth')

    #todo:可视化得分、类别、框
    val(epoch, mymodel, val_loader, mycriterion, mydevice)

writer.close()
wandb.finish()
torch.cuda.empty_cache()