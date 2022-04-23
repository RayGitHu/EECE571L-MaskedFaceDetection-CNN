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

model_dir='modelWithBestValidationAcc_unmasked_100epochs_v2.h5'
root_unmasked='dataset/unmasked_test'
root_masked='dataset/masked_test'
classifier_path='classification_model.h5'

#the function you need to use to remove the classifier layer
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def init_weights(m):
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

def preprocess(image):

    masked  =  image / 255 

    for ch in range (0,3):
        masked[:,:,ch]=(masked[:,:,ch]-meanDataset[ch])/stdDataset[ch]
        
    masked=ResizedTensor(masked,(256,256))
    masked=CenterCropTensor(masked,(224,224))


    masked=torch.tensor(masked).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)  #this part move the image from CPU to GPU
    return masked

## train data
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
                 
## test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class BinaryClassification(nn.Module): #basebone for the weighting model
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
def binary_acc(y_pred, y_test): #to calculate the accuracy
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meanDataset =[0.4415734549019608,0.388225627197976,0.3462771010120177]
stdDataset =[0.2711501212166684,0.24437697540137235,0.2350548283233844]  

#create the dataset
dataset_dic={'unmasked_image_path':[],'masked_positive_path':[],'masked_negative_path':[]} # masked positive doesn't include the same masked image of the input unmasked image
unmasked_images=os.listdir(root_unmasked)
masked_images=os.listdir(root_masked)
for i in range(len(unmasked_images)):
    dataset_dic['unmasked_image_path'].append(root_unmasked+'/'+unmasked_images[i])
    pos,neg=[],[]
    for j in range(len(masked_images)):
        if (unmasked_images[i].split('_')[0]+'_'+unmasked_images[i].split('_')[1]== masked_images[j].split('_')[0]+'_'+masked_images[j].split('_')[1] ):
            pos.append(root_masked+'/'+masked_images[j])
        elif (unmasked_images[i].split('_')[0]+'_'+unmasked_images[i].split('_')[1]!= masked_images[j].split('_')[0]+'_'+masked_images[j].split('_')[1]):
            neg.append(root_masked+'/'+masked_images[j])
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
    for i in range(len(dataset_dic['unmasked_image_path'])):
        ff,ff_p,ff_n=[],[],[]
        #print(dataset_dic['unmasked_image_path'][i])
        unmasked_imagee=cv2.imread(dataset_dic['unmasked_image_path'][i])
        processed_unmasked_image=preprocess(unmasked_imagee)
        f=model(processed_unmasked_image).cpu().detach().numpy()
        f_norm=f/np.sqrt(np.dot(f,np.transpose(f)))
        if len(dataset_dic['masked_positive_path'][i])!=0:
            for j in range(len(dataset_dic['masked_positive_path'][i])):
                #print(dataset_dic['masked_positive_path'][i][j])
                masked_image_pos=cv2.imread(dataset_dic['masked_positive_path'][i][j])
                processed_masked_image_pos=preprocess(masked_image_pos)
                f_p=model(processed_masked_image_pos).cpu().detach().numpy()
                f_p_norm=f_p/np.sqrt(np.dot(f_p,np.transpose(f_p)))
                A=f_norm-f_p_norm
                dic={}
                for kk in range(len(A[0])):
                    dic[str(kk+1)]=A[0][kk]
                dic['s_d']='same'
                a=dataset_dic['unmasked_image_path'][i].split('/')[2].split('_')
                dic['unmasked_name']=a[0]+' '+a[1]
                data.update({str(count):dic})
                count+=1
        if len(dataset_dic['masked_negative_path'][i])!=0:
            for k in range(len(dataset_dic['masked_negative_path'][i])):
                masked_image_neg=cv2.imread(dataset_dic['masked_negative_path'][i][k])
                processed_masked_image_neg=preprocess(masked_image_neg)
                f_n=model(processed_masked_image_neg).cpu().detach().numpy()
                f_n_norm=f/np.sqrt(np.dot(f_n,np.transpose(f_n)))
                B=f_norm-f_n_norm
                dic={}
                for kkk in range(len(B[0])):
                    dic[str(kkk+1)]=B[0][kkk]
                dic['s_d']='different'
                a=dataset_dic['unmasked_image_path'][i].split('/')[2].split('_')
                dic['unmasked_name']=a[0]+' '+a[1]
                data.update({str(count):dic})
                count+=1
df=pd.DataFrame(data).transpose()
#df = shuffle(df)
df['s_d'] = df['s_d'].astype('category')
encode_map = {
    'same': 1,
    'different': 0
}
df['s_d'].replace(encode_map, inplace=True)
# condition with df.values property
'''same= df['s_d'].values == 1
different= df['s_d'].values == 0
# new dataframe
df_same = df[same]
df_different1=df[different]
df_different=df_different1.iloc[:1596,:] # this is to create a balanced dataset since we had 1596 cases of the "same" class, we got 1596 of the "different" class and combine them
frames=[df_same,df_different]
result = pd.concat(frames)''' 
#preprocessing the data
result=df
X_test = result.iloc[:, 0:-2]
s_d = result.iloc[:, -2:-1]
un_names = result.iloc[:, -1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
scaler = pickle.load(open('scaler.pkl', 'rb'))
#X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#train_data = TrainData(torch.FloatTensor(X_train), 
#                       torch.FloatTensor(y_train))
## test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
            
test_data = TestData(torch.FloatTensor(X_test))
#train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)
classifier=torch.load(classifier_path)
classifier.to(device).eval()
i=0
a=[]
with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = classifier(X_batch)
            #y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            
            if y_pred_tag.item()==1:
                a.append(np.array(un_names)[i])
                print(np.array(un_names)[i])
                #print('same')
#            else:
#                print('different')
            i+=1
