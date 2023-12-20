import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from model import HRFNet
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import datetime, yaml
from tqdm import tqdm
from utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tb = SummaryWriter()
from dataloader import CustomDataset
set_random_seed(1221)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'

dataset = CustomDataset('data/train.txt' )
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

dataset = CustomDataset('data/valid.txt' )
val_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = HRFNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
criterion = nn.CrossEntropyLoss()

max_val_auc = 0
max_val_iou = [0.0, 0.0]

model.train()
for epoch in range(50):
    train_loss = AverageMeter()
    train_inter = AverageMeter()
    train_union = AverageMeter()

    for sample in tqdm(train_loader):
        
        optimizer.zero_grad()

        img = sample[0].to(device)
        tar = sample[1].to(device)

        pred = model(img)

        loss = criterion(pred, tar.long().detach())
        loss.backward()
        optimizer.step()

        train_loss.update(loss.detach().cpu().item())

        intr, uni = batch_intersection_union(pred, tar, num_class = 2)

        train_inter.update(intr)
        train_union.update(uni)
    
    scheduler.step() 
    train_softmax = train_loss.avg
        
    train_IoU = train_inter.sum/(train_union.sum + 1e-10)
    train_IoU = train_IoU.tolist()
    train_mIoU = np.mean(train_IoU)
    train_mIoU = train_mIoU.tolist()


    with torch.no_grad():
        model.eval()
        val_inter = AverageMeter()
        val_union = AverageMeter()
        val_pred = []
        val_tar = []
        auc = []
        for img, tar in tqdm(val_loader):
            img, tar = img.to(device), tar.to(device)
            pred= model(img)
            intr, uni = batch_intersection_union(pred, tar, num_class =2)
            val_inter.update(intr)
            val_union.update(uni)
            
            y_score = F.softmax(pred, dim=1)[:,1,:,:]
            
            # the following auc code is taken from:
            # https://github.com/ZhiHanZ/IRIS0-SPAN/blob/main/utils/metrics.py
            
            for yy_true, yy_pred in zip(tar.cpu().numpy(), y_score.cpu().numpy()):
                this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
                that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel())
                auc.append(max(this, that))

        val_auc = np.mean(auc)

        val_pred = []
        val_tar = []

        if val_auc > max_val_auc:
            max_val_auc = val_auc

        val_IoU = val_inter.sum/(val_union.sum + 1e-10)
        val_IoU = val_IoU.tolist()
        val_mIoU = np.mean(val_IoU)
        val_mIoU = val_mIoU.tolist()

        if val_IoU[1] > max_val_iou[1]:
            max_val_iou = val_IoU

        logs = {'epoch': epoch, 'Softmax Loss':train_softmax,
        'Train IoU':train_IoU, 'Validation IoU': val_IoU, 'Validation AUC': val_auc, 
        'Max Validaton_AUC': max_val_auc, "Max IoU Tampered": max_val_iou}

        tb.add_scalar("auc", val_auc, epoch+1)
