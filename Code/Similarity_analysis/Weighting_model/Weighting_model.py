import torch
import os
import cv2
from tqdm.notebook import tqdm
import cv2 
import torch
import argparse
import os
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
import random
from datetime import datetime
import xlsxwriter
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#compensate for data imbalance
from sklearn.utils import shuffle
import pickle


EPOCHS =20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
root_unmasked='dataset/unmasked_attached'
root_masked='dataset/masked_attached'
model_dir='modelWithBestValidationAcc_unmasked_100epochs_v2.h5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meanDataset =[0.4415734549019608,0.388225627197976,0.3462771010120177] # mean value of the training dataset which will be needed to normalize the images
stdDataset =[0.2711501212166684,0.24437697540137235,0.2350548283233844] # standard deviation of the training dataset which will be needed to normalize the images

#the function you need to use to remove the classifier layer
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def init_weights(m): #initial weights that are used to train the model
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


def ResizedTensor(image, size): # the function to resize the images to the desired size

    dim=size
    resized = cv2.resize(image, dim, interpolation =  cv2.INTER_LINEAR )
    #print(resized.shape)
    return resized

def CenterCropTensor(image, size): # the function to to do center crop

    image_width, image_height, channel = image.shape

    #image_width, image_height = _get_image_size(img)
    crop_height, crop_width = size

    # crop_top = int(round((image_height - crop_height) / 2.))
    # Result can be different between python func and scripted func
    # Temporary workaround:
    crop_top = int((image_height - crop_height + 1) * 0.5)
    # crop_left = int(round((image_width - crop_width) / 2.))
    # Result can be different between python func and scripted func
    # Temporary workaround:
    crop_left = int((image_width - crop_width + 1) * 0.5)
    img=image[ crop_left: crop_left+crop_width,crop_top:crop_top+crop_height,:];
    return img #crop(img, crop_top, crop_left, crop_height, crop_width)

def preprocess(image): #pre-processing steps to prepare the image to be fed into our model which includes normalization, resizing, crop center, and changing the format into tensor

    masked  =  image / 255 

    for ch in range (0,3):
        masked[:,:,ch]=(masked[:,:,ch]-meanDataset[ch])/stdDataset[ch]
        
    masked=ResizedTensor(masked,(256,256))
    masked=CenterCropTensor(masked,(224,224))


    masked=torch.tensor(masked).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)  #this part move the image from CPU to GPU
    return masked

## create a class of train data
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
                 
## create a class of test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class BinaryClassification(nn.Module): # the basbone of our weighting model
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(4096, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs)) 
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        #x = self.dropout(x)
        x = self.layer_out(x) 
        x=torch.sigmoid(x)
        
        return x
def binary_acc(y_pred, y_test): # afunction to calculate the accuracy
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc
    
    #create the dataset
dataset_dic={'unmasked_image_path':[],'masked_positive_path':[],'masked_negative_path':[]} # masked positive doesn't include the same masked image of the input unmasked image
unmasked_images=os.listdir(root_unmasked)
masked_images=os.listdir(root_masked)

for i in range(len(unmasked_images)): #the directory of all the unmasked images
    dataset_dic['unmasked_image_path'].append(root_unmasked+'/'+unmasked_images[i])
    pos,neg=[],[]
    for j in range(len(masked_images)): #the directory of all the masked images
        if (unmasked_images[i].split('_')[0]+'_'+unmasked_images[i].split('_')[1]== masked_images[j].split('_')[0]+'_'+masked_images[j].split('_')[1] and unmasked_images[i]!=masked_images[j][:-9] ):
            pos.append(root_masked+'/'+masked_images[j]) # if the masked image has the same identity as the unmasked image
        elif (unmasked_images[i].split('_')[0]+'_'+unmasked_images[i].split('_')[1]!= masked_images[j].split('_')[0]+'_'+masked_images[j].split('_')[1]):
            neg.append(root_masked+'/'+masked_images[j]) # if the masked image does not have the same identity as the unmasked image
    dataset_dic['masked_positive_path'].append(pos)
    dataset_dic['masked_negative_path'].append(neg)

#model with no classifier (feature extractor)
model=torch.load(model_dir).eval()
model.classifier[6]=Identity()
model.fc=Identity()
model.to(device).eval() #put the model on GPU

data={} # this is the final dictionay of the dataset 
count=1
if True:
    for i in range(100):
        ff,ff_p,ff_n=[],[],[]
        #print(dataset_dic['unmasked_image_path'][i])
        unmasked_imagee=cv2.imread(dataset_dic['unmasked_image_path'][i]) #read each image
        processed_unmasked_image=preprocess(unmasked_imagee) # prepare each image
        f=model(processed_unmasked_image).cpu().detach().numpy() # extract the feature vector of each image
        f_norm=f/np.sqrt(np.dot(f,np.transpose(f))) #normalize the feature vector
        for j in range(5):
             #print(dataset_dic['masked_positive_path'][i][j])
             masked_image_pos=cv2.imread(dataset_dic['masked_positive_path'][i][j])#read each image
             processed_masked_image_pos=preprocess(masked_image_pos)# prepare each image
             f_p=model(processed_masked_image_pos).cpu().detach().numpy()# extract the feature vector of each image
             f_p_norm=f_p/np.sqrt(np.dot(f_p,np.transpose(f_p)))#normalize the feature vector
             A=f_norm-f_p_norm #calculate the error between the feature vectors
             dic={}
             for kk in range(len(A[0])):
                dic[str(kk+1)]=A[0][kk]
             dic['s_d']='same' 
             data.update({str(count):dic})
             count+=1

        for k in range(5):
             masked_image_neg=cv2.imread(dataset_dic['masked_negative_path'][i][k])
             processed_masked_image_neg=preprocess(masked_image_neg)
             f_n=model(processed_masked_image_neg).cpu().detach().numpy()
             f_n_norm=f/np.sqrt(np.dot(f_n,np.transpose(f_n)))
             B=f_norm-f_n_norm
             dic={}
             for kkk in range(len(B[0])):
                dic[str(kkk+1)]=B[0][kkk]
             dic['s_d']='different'
             data.update({str(count):dic})
             count+=1

df=pd.DataFrame(data).transpose() #create a data frame of the dictionay of the dataset
df = shuffle(df)

df['s_d'] = df['s_d'].astype('category')
encode_map = {
    'same': 1,
    'different': 0
}

df['s_d'].replace(encode_map, inplace=True)

#preprocessing the data
result=shuffle(df) # randomize the order of the data
X = result.iloc[:, 0:-1]
y = result.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69) #split the data into train and test with ratio of 0.33

#scaler = StandardScaler() 
#X_train = scaler.fit_transform(X_train) 
scaler = pickle.load(open('scaler.pkl', 'rb'))
X_train = scaler.transform(X_train)# normalize the data based on the training dataset
X_test = scaler.transform(X_test) # normalize the data based on the training dataset

train_data = TrainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))
      
test_data = TestData(torch.FloatTensor(X_test))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True) #create batches of dataset and shuffle the dataset
test_loader = DataLoader(dataset=test_data, batch_size=1)

#weighting model
model = BinaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for e in range(1, EPOCHS+1): #this for loop goes through each epoch and train the model and update the weights after each epoch
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1)) # find the error between the prediction and groundtruth 
        acc = binary_acc(y_pred, y_batch.unsqueeze(1)) #find the accuracy of the prediction
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader: # evaluate the accuracy of the prediction in unseen test dataset
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        #y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix(y_test, y_pred_list)
print(classification_report(y_test, y_pred_list))

torch.save(model,'classification_model.h5') #save the trained model
