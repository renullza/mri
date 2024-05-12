import torch
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
import time
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

# end of imports

PATH = "/weights/model_state_dict.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using gpu")
else:
    print("using cpu")

bs_train, bs_val, bs_test = 16, 8, 1
epochs = 100
lr = 0.0001

train_files = []
mask_files = glob('D:/projects/mri/kaggle/lgg-mri-segmentation/kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask',''))

#sample plot
# rows,cols=3,3
# fig=plt.figure(figsize=(10,10))
# for i in range(1,rows*cols+1):
#     fig.add_subplot(rows,cols,i)
#     img_path=train_files[i]
#     msk_path=mask_files[i]
#     img=cv2.imread(img_path)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     msk=cv2.imread(msk_path)
#     plt.imshow(img)
#     plt.imshow(msk,alpha=0.4)
# plt.show()

#Create dataframes with paths for training, validation, and testing
df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
df_train, df_test = train_test_split(df,test_size = 0.1)
df_train, df_val = train_test_split(df_train,test_size = 0.2)

#Create a custom dataset class which applies identical transformations to the training and validation data
class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)

#Datasets
train_dataset = MyDataset(df_train["filename"].values.tolist(), df_train["mask"].values.tolist())
val_dataset = MyDataset(df_val['filename'].values.tolist(), df_val['mask'].values.tolist())
test_dataset = MyDataset(df_test['filename'].values.tolist(), df_test["mask"].values.tolist())

#Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=bs_val, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs_test, shuffle=True)

#Dice loss function 
# read https://www.linkedin.com/pulse/in-depth-exploration-loss-functions-deep-learning-kiran-dev-yadav/ for better loss function
def dc_loss(pred, target):
    smooth = 100

    predf = pred.view(-1)
    targetf = target.view(-1)
    intersection = (predf * targetf).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (predf.sum() + targetf.sum() + smooth))

#Define the UNet architecture


def conv_layer(input_channels, output_channels):     #This is a helper function to create the convolutional blocks
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_1 = conv_layer(3, 64)
        self.down_2 = conv_layer(64, 128)
        self.down_3 = conv_layer(128, 256)
        self.down_4 = conv_layer(256, 512)
        self.down_5 = conv_layer(512, 1024)
        
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = conv_layer(1024, 512)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = conv_layer(512, 256)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = conv_layer(256, 128)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = conv_layer(128, 64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        self.output_activation = nn.Sigmoid()
                
    def forward(self, img):     #The print statements can be used to visualize the input and output sizes for debugging
        x1 = self.down_1(img)
        #print(x1.size())
        x2 = self.max_pool(x1)
        #print(x2.size())
        x3 = self.down_2(x2)
        #print(x3.size())
        x4 = self.max_pool(x3)
        #print(x4.size())
        x5 = self.down_3(x4)
        #print(x5.size())
        x6 = self.max_pool(x5)
        #print(x6.size())
        x7 = self.down_4(x6)
        #print(x7.size())
        x8 = self.max_pool(x7)
        #print(x8.size())
        x9 = self.down_5(x8)
        #print(x9.size())
        
        x = self.up_1(x9)
        #print(x.size())
        x = self.up_conv_1(torch.cat([x, x7], 1))
        #print(x.size())
        x = self.up_2(x)
        #print(x.size())
        x = self.up_conv_2(torch.cat([x, x5], 1))
        #print(x.size())
        x = self.up_3(x)
        #print(x.size())
        x = self.up_conv_3(torch.cat([x, x3], 1))
        #print(x.size())
        x = self.up_4(x)
        #print(x.size())
        x = self.up_conv_4(torch.cat([x, x1], 1))
        #print(x.size())
        
        x = self.output(x)
        x = self.output_activation(x)
        #print(x.size())
        
        return x

#Initialize the model and optimizer 
#develope  this section later
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))

#Load a preexisting set of weights if continuting training
model.load_state_dict(torch.load(PATH))
print('doom')