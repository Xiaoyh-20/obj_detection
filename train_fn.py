"""
File: train_fn.py
Author: xiaoyihong
Created on: 2023/12/24
Description: 
    封装单次训练函数，验证函数，测试函数
"""
import numpy as np
import torch
import time

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from PIL import Image, ImageDraw, ImageFont

import wandb
from data_process_fn import criterion_mAP


wandb.init(project="maskrcnn_demo")
wandb.define_metric("epoch")
wandb.define_metric("train_loss", step_metric="epoch")
wandb.define_metric("val_loss", step_metric="epoch")
wandb.define_metric("mAP", step_metric="epoch")


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

font = ImageFont.load_default(size=25)

def train(epoch, model, train_loader, optimizer,device):
    writer = SummaryWriter("./runs/train")
    model.train()
    total_loss = 0
    loss_classifier_per_epoch=0
    loss_box_reg_per_epoch=0
    loss_mask_per_epoch=0
    loss_objectness_per_epoch=0
    loss_rpn_box_reg_per_epoch=0

    start_time=time.time()
    batch_idx=0
    for images, targets in train_loader:
        #可视化训练集输入
        for idx, image in enumerate(images):
            img = torch.tensor(image * 255, dtype=torch.uint8)
            boxes = targets[idx]["boxes"]
            # 如果 boxes 是一维的，添加一个额外的维度
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)
            img_box = draw_bounding_boxes(img, boxes, width=5)

            img_pil=Image.fromarray(img_box.permute(1,2,0).cpu().numpy())
            draw=ImageDraw.Draw(img_pil)

            for box, label in zip(targets[idx]["boxes"], targets[idx]["labels"]):
                draw.text((box[0], box[1]), str(classes[label.item()]), fill="white", font=font)
            img_box_label=torch.tensor(np.array(img_pil)).permute(2,0,1)
            writer.add_image(f"train_images_{epoch}", img_box_label, idx + 4 * (batch_idx))

            mask_bool = (targets[idx]["masks"] > 0).squeeze(1)
            masked_image = draw_segmentation_masks(torch.tensor(image * 255, dtype=torch.uint8).to("cpu"),
                                                   mask_bool.to("cpu"))  # debug:注意 此处 color 也是一个张量，需要送到同一个 device 上
            writer.add_image(f"train_masked_images{epoch}", masked_image, idx + 4 * (batch_idx))
        batch_idx+=1

        images=[image.to(device) for image in images] #debug:list不能直接to(device)
        targets=[{k:v.to(device) for k,v in target.items()} for target in targets]#debug:dict不能直接to(device)

        optimizer.zero_grad()

        loss_dict = model(images, targets)#debug:需要同时输入input和target
        loss_classifier_per_epoch += loss_dict["loss_classifier"].item() * len(images)
        loss_box_reg_per_epoch += loss_dict["loss_box_reg"].item() * len(images)
        loss_objectness_per_epoch += loss_dict["loss_objectness"].item() * len(images)
        loss_rpn_box_reg_per_epoch += loss_dict["loss_rpn_box_reg"].item() * len(images)

        train_loss=(loss_dict["loss_classifier"]+loss_dict["loss_box_reg"]+loss_dict["loss_objectness"]+loss_dict["loss_rpn_box_reg"])*len(images)
        train_loss.backward()

        total_loss+=train_loss.item()*len(images)#debug:注意这里的images是一个list



        optimizer.step()


        wandb.log({
            "loss_classifier": loss_dict["loss_classifier"].item(),
            "loss_box_reg": loss_dict["loss_box_reg"].item(),
            "loss_objectness": loss_dict["loss_objectness"].item(),
            "loss_rpn_box_reg": loss_dict["loss_rpn_box_reg"].item(),
        })

    loss_classifier_per_epoch = loss_classifier_per_epoch/len(train_loader.dataset)
    loss_box_reg_per_epoch = loss_box_reg_per_epoch/len(train_loader.dataset)
    loss_objectness_per_epoch = loss_objectness_per_epoch/len(train_loader.dataset)
    loss_rpn_box_reg_per_epoch = loss_rpn_box_reg_per_epoch/len(train_loader.dataset)

    wandb.log({
        "train_loss":total_loss/len(train_loader.dataset),
        "loss_classifier_per_epoch": loss_classifier_per_epoch,
        "loss_box_reg_per_epoch": loss_box_reg_per_epoch,
        "loss_objectness_per_epoch": loss_objectness_per_epoch,
        "loss_rpn_box_reg_per_epoch": loss_rpn_box_reg_per_epoch
    })

    end_time=time.time()

    print(f"*****************epoch {epoch}*********************\n train_loss = {total_loss/len(train_loader.dataset)}\t  time = {end_time-start_time}\n")

def val(epoch, model, val_loader, criterion, device):
    writer = SummaryWriter("./runs/train")
    val_loss = 0
    preds_all=[]
    gt_all=[]
    start_time=time.time()
    print(f"***val***\n")
    with torch.no_grad():
        batch_idx=0

        for images, targets in val_loader:
             #可视化验证集输入
            for idx, image in enumerate(images):
                img=torch.tensor(image*255, dtype=torch.uint8)
                img_box=draw_bounding_boxes(img, targets[idx]["boxes"], width=5)

                img_pil=Image.fromarray(img_box.permute(1,2,0).cpu().numpy())
                draw=ImageDraw.Draw(img_pil)

                for box, label in zip(targets[idx]["boxes"], targets[idx]["labels"]):
                    draw.text((box[0], box[1]), str(classes[label.item()]), fill="white", font=font)
                img_box_label=(torch.tensor(np.array(img_pil))).permute(2,0,1)
                writer.add_image(f"val_images_{epoch}", img_box_label, idx+4*(batch_idx))
            batch_idx+=1

            images=[image.to(device) for image in images]
            targets=[{k:v.to(device) for k,v in target.items()} for target in targets]

            model.train()
            pred_loss=model(images, targets)
            #todo：修改损失函数
            val_loss+=(pred_loss["loss_classifier"].item()+pred_loss["loss_box_reg"].item()+pred_loss["loss_objectness"].item()+pred_loss["loss_rpn_box_reg"].item())*len(images)

            model.eval()
            # print("eval:start")
            preds = model(images)
            #可视化目标检测结果
            for idx, image in enumerate(images):
                img = torch.tensor(image * 255, dtype=torch.uint8)
                img_box = draw_bounding_boxes(img, preds[idx]["boxes"], width=5)

                img_pil=Image.fromarray(img_box.permute(1,2,0).cpu().numpy())
                draw=ImageDraw.Draw(img_pil)
                # print(preds[idx]["labels"])
                for box, label in zip(preds[idx]["boxes"], preds[idx]["labels"]):
                    draw.text((box[0], box[1]), str(classes[label.item()]), fill="white", font=font)
                img_box_label=torch.tensor(np.array(img_pil)).permute(2,0,1)
                writer.add_image(f"val_result_{epoch}", img_box_label, idx+4*(batch_idx))

            preds_all.append(preds)
            gt_all.append(targets)# 数据类型：[[{}]]
    end_time=time.time()
    val_loss=val_loss/len(val_loader.dataset)
    mAP=criterion(gt_all, preds_all, num_classes=21, tp_thres=0.7, fp_thres=0.3)

    wandb.log({
        "val_loss":val_loss,
        "mAP":mAP
    })
    print(f"val_loss:{val_loss}\t mAP: {mAP}\t time: {end_time-start_time}\n")

def test(epoch, model, test_loader, criterion, device):
    writer = SummaryWriter("./runs/test")
    preds_all=[]
    gt_all = []
    start_time = time.time()
    print(f" **************epoch: {epoch}*****************\n")
    with torch.no_grad():
        batch_idx = 0

        for images, targets in test_loader:
            # 可视化验证集输入
            for idx, image in enumerate(images):
                img = torch.tensor(image * 255, dtype=torch.uint8)
                img_box = draw_bounding_boxes(img, targets[idx]["boxes"], width=5)

                img_pil = Image.fromarray(img_box.permute(1, 2, 0).cpu().numpy())
                draw = ImageDraw.Draw(img_pil)

                for box, label in zip(targets[idx]["boxes"], targets[idx]["labels"]):
                    draw.text((box[0], box[1]), str(classes[label.item()]), fill="white", font=font)
                img_box_label = (torch.tensor(np.array(img_pil))).permute(2, 0, 1)
                writer.add_image(f"val_images_{epoch}", img_box_label, idx + 4 * (batch_idx))
            batch_idx += 1

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            model.eval()
            preds = model(images)
            # 可视化目标检测结果
            for idx, image in enumerate(images):
                img = torch.tensor(image * 255, dtype=torch.uint8)
                img_box = draw_bounding_boxes(img, preds[idx]["boxes"], width=5)

                img_pil = Image.fromarray(img_box.permute(1, 2, 0).cpu().numpy())
                draw = ImageDraw.Draw(img_pil)

                # print(preds[idx]["labels"])

                for box, label in zip(preds[idx]["boxes"], preds[idx]["labels"]):
                    draw.text((box[0], box[1]), str(classes[label.item()]), fill="white", font=font)
                img_box_label = torch.tensor(np.array(img_pil)).permute(2, 0, 1)
                writer.add_image(f"val_result_{epoch}", img_box_label, idx + 4 * (batch_idx))

            preds_all.append(preds)
            gt_all.append(targets)  # 数据类型：[[{}]]
    end_time=time.time()

    mAP = criterion(gt_all, preds_all, num_classes=21, tp_thres=0.7, fp_thres=0.3)
    print(f"test finished\t mAP:{mAP}\t time: {end_time-start_time}\n")