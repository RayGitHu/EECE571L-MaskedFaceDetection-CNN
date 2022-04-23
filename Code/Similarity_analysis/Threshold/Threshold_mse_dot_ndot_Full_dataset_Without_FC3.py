

#Modified by Hamid Reza Tohidypour for DML of UBC
#Confidential do not distribute it beyond your team of ECE571


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


workbook = xlsxwriter.Workbook('Threshold.xlsx')
worksheet = workbook.add_worksheet()
# Widen the first column to make the text clearer.
worksheet.set_column('A:A', 20)

# Add a bold format to use to highlight cells.
bold = workbook.add_format({'bold': True})
worksheet.write('A1', 'Identity', bold)
worksheet.write('B1', 'MSE', bold)
worksheet.write('C1', 'Dot', bold)
worksheet.write('D1', 'Dot_normalized', bold)




parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
parser.add_argument("--LocationofSavedModel", type=str, required=True, help="Path to checkpoint model")
opt = parser.parse_args()
print(opt)


############ mean and SD of the train dataset (Mixed dataset)
meanDataset = [0.4706155413031573, 0.4155981526458318, 0.3794127018972444]
stdDataset = [0.2681391062637582 ,0.24723137207986798, 0.24271453938450466]


#the function you need to use to remove the classifier layer
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the model without the classifer layer
feature_extractror= torch.load(opt.LocationofSavedModel).eval()
print(feature_extractror)
#extract the layer before the classifier layer
feature_extractror.classifier[6]=Identity()
feature_extractror.fc=Identity()
print(feature_extractror)



'''
# load the model with the classifer layer
feature_extractror= torch.load(opt.LocationofSavedModel).eval()
feature_extractror.to(device).eval() #put the model on GPU
'''

############ just to calculate the execution time
tm = datetime.now().strftime("%Y%m%d%H%M%S")
start = datetime.now()
dataset_path= opt.dataset_path






############  the for loop that reads the image inside the folder
row=1
identities=os.listdir(dataset_path+'/'+'Unmasked')
for identity in identities:
    mse,Dot,Dot_normalized=[],[],[]    
    if identity !='.DS_Store':
        images_unmasked=os.listdir(dataset_path+'/'+'Unmasked'+'/'+identity)
        images_masked=os.listdir(dataset_path+'/'+'Masked'+'/'+identity)
        for i in range(len(images_unmasked)):
            with torch.no_grad():
                unmasked_path=dataset_path+'/'+'Unmasked'+'/'+identity+'/'+images_unmasked[i]
                unmasked = cv2.imread(unmasked_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) #read the image
                unmasked  =  unmasked  / 255
                for ch in range (0,3):
                        unmasked[:,:,ch]=(unmasked[:,:,ch]-meanDataset[ch])/stdDataset[ch]
                unmasked=ResizedTensor(unmasked,(256,256))
                unmasked=CenterCropTensor(unmasked,(224,224))
                unmasked=torch.tensor(unmasked).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)  #this part move the image from CPU to GPU
                features_unmasked = feature_extractror(unmasked)  #get the output of the model
                np_features_unmasked  =  features_unmasked.cpu().numpy() 
                #print (np_features_unmasked)  #this is the feature we want to compare

                for j in range(len(images_masked)):
                    masked_path=dataset_path+'/'+'Masked'+'/'+identity+'/'+images_masked[j]
                    masked=cv2.imread(masked_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) #read the image
                    masked  =  masked  / 255 
                    
                    for ch in range (0,3):
                        masked[:,:,ch]=(masked[:,:,ch]-meanDataset[ch])/stdDataset[ch]
                        
                    masked=ResizedTensor(masked,(256,256))
                    masked=CenterCropTensor(masked,(224,224))
                    
                    
                    masked=torch.tensor(masked).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)  #this part move the image from CPU to GPU
                    
                    features_masked = feature_extractror(masked)  #get the output of the model
                    
                    np_features_masked  =  features_masked.cpu().numpy() 

                    # the Euclidean norm of a vector can be defined as the square root of the inner product of a vector with itself.
                    np_features_masked_normalized=np_features_masked/np.sqrt(np.dot(np_features_masked,np.transpose(np_features_masked)))
                    np_features_unmasked_normalized=np_features_unmasked/np.sqrt(np.dot(np_features_unmasked,np.transpose(np_features_unmasked)))
                    
                    #print (np_features_masked)  #this is the feature we want to compare
                    mse.append(  np.sum(np.abs(np_features_masked-np_features_unmasked))/len(np_features_masked[0]) )
                    Dot.append(np.dot(np_features_masked,np.transpose(np_features_unmasked)))
                    Dot_normalized.append(np.dot(np_features_masked_normalized,np.transpose(np_features_unmasked_normalized)))
                    
        worksheet.write(row,0,identity )
        worksheet.write(row,1,np.mean(np.array(mse)))
        worksheet.write(row,2,np.mean(np.array(Dot)))
        worksheet.write(row,3,np.mean(np.array(Dot_normalized)))
        row+=1
        
workbook.close()                    

print("Total running time:", datetime.now() - start)

