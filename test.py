
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

set_random_seed(1221)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'

img_shape = 1000
# Define a custom dataset
# Define the transformations
transformations = transforms.Compose([
    transforms.Resize((img_shape,img_shape )),  # Resize the image to 1000x1000
    # Normalize using ImageNet mean and std
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RandomImageDataset(Dataset):
    def __init__(self, num_images, transform=None):
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Generate a random image (3 channels, with random values)
        # We'll use a smaller size initially for memory efficiency
        image = np.random.rand(224, 224, 3).astype('float32')  # Corrected shape
        
        # Convert numpy array to PIL Image
        image = Image.fromarray((image * 255).astype('uint8'), 'RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Dummy label for the sake of completeness
        label = torch.rand(img_shape,img_shape)  #torch.tensor(0, dtype=torch.long)
        label =  (label >= 0.5).int()
        
        return image, label

# Create the dataset
dataset = RandomImageDataset(num_images=100, transform=transformations)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
train_loader = dataloader
val_loader = dataloader
# Check out what's inside the DataLoader
# train_features_batch, train_labels_batch = next(iter(dataloader))
# print(train_features_batch.shape, train_labels_batch.shape)

model = HRFNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
criterion = nn.CrossEntropyLoss()

max_val_auc = 0
max_val_iou = [0.0, 0.0]

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
