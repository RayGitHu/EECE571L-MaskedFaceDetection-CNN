
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


# Threshold values using the MSE DOT and DOT_NORM approaches

		
MSE_treshold =0.216870382   # without classifier (full dataset)
DOT_threshold = 1029.688143   # without classifier (full dataset)
DOT_NORM_threshold = 0.68939216   # without classifier (full dataset)

'''		
MSE_treshold =3.209509571	  # with classifier (full dataset)
DOT_threshold = 7485.986052   # with classifier (full dataset)
DOT_NORM_threshold =0.968664979   # with classifier (full dataset)
'''

'''		
MSE_treshold =0.239562387    # without classifier (removed dataset)
DOT_threshold = 948.3677224     # without classifier (removed dataset)
DOT_NORM_threshold = 0.647773446	  # without classifier (removed dataset)
'''


'''	
MSE_treshold =3.478223226   # with classifier (removed dataset)
DOT_threshold = 7357.339471	  # with classifier (removed dataset)
DOT_NORM_threshold = 0.965166087	  # with classifier (removed dataset)
'''



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



############  the for loop that reads the images inside the folders
# The number of times the classification happens
count=0
# The number of times the classification was correct using the MSE approach
correct_MSE=0
# The number of times the classification was correct using the DOT approach
correct_DOT=0
# The number of times the classification was correct using the DOT_NORM approach
correct_DOT_NORM=0



identities=sorted (os.listdir(dataset_path+'/'+'Unmasked'))
for identity in identities:
    

    if identity !=".DS_Store":
        images_masked=os.listdir(dataset_path+'/'+'Unmasked'+'/'+identity)
        
        #images_masked=os.listdir(dataset_path+'/'+'masked_dataset_FE'+'/'+identity)
        
        for i in range(len(images_masked)):
            with torch.no_grad():
                
                # Reading the image
                if images_masked[i] !=".DS_Store":
                    
                    masked_path=dataset_path+'/'+'Unmasked'+'/'+identity+'/'+images_masked[i]
                    masked = cv2.imread(masked_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

                    # Normalization step
                    masked  =  masked  / 255
                    for ch in range (0,3):
                            masked[:,:,ch]=(masked[:,:,ch]-meanDataset[ch])/stdDataset[ch]

                    # Resizing the image and center crop same as the validation step
                    masked=ResizedTensor(masked,(256,256))
                    masked=CenterCropTensor(masked,(224,224))

                    # Moving the image from CPU to GPU
                    masked=torch.tensor(masked).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)  

                    # Getting the output of the model
                    features_masked = feature_extractror(masked)  
                    # Converting tensor to numpy
                    np_features_masked  =  features_masked.cpu().numpy() 
                    #print (np_features_masked)  #this is the feature we want to compare

                    # counter to keep track of the number of times the classification happens (equal to number of masked images)
                    count+=1
                    print ("COUNT: ",count)




                    # Boolean varaible to check if we the two images belong to the same identity 
                    same= False   
                    similar_mse=[]
                    different_mse=[]

                    similar_dot=[]
                    different_dot=[]

                    similar_ndot=[]
                    different_ndot=[]

                   
                    
                    # Rearranging list starting from a certain element
                    index= identities.index(identity)
                    identities_=identities[index:] + identities[:index]
                    
                   

                    MSE_value_reference=[]
                    DOT_value_reference=[]
                    DOT_NORM_value_reference=[]
                    for identity_ in identities_:
                        if identity_ !=".DS_Store":
                            # List of the names of the unmasked images for the current identity
                            images_unmasked=os.listdir(dataset_path+'/'+'Masked'+'/'+identity_)
                            if identity_ == identity:
                                same= True  
                                print ("SAME IDENITIY")
                            else :
                                same= False
                                print ("DIFFERENT IDENITIY")


                            
                            # Looping through the unmasked images of the current identity 
                            for j in range(len(images_unmasked)):
                                    # Reading the image 
                                    if images_unmasked[j] !=".DS_Store":
                                        unmasked_path=dataset_path+'/'+'Masked'+'/'+identity_+'/'+images_unmasked[j]
                                        unmasked=cv2.imread(unmasked_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 

                                        # Normalization step
                                        unmasked  =  unmasked  / 255 
                                        for ch in range (0,3):
                                            unmasked[:,:,ch]=(unmasked[:,:,ch]-meanDataset[ch])/stdDataset[ch]

                                        # Resizing the image and center crop same as the validation step    
                                        unmasked=ResizedTensor(unmasked,(256,256))
                                        unmasked=CenterCropTensor(unmasked,(224,224))

                                        # Moving the image from CPU to GPU
                                        unmasked=torch.tensor(unmasked).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float)  

                                        # Getting the output of the model
                                        features_unmasked = feature_extractror(unmasked)  #get the output of the model
                                        # Converting tensor to numpy
                                        np_features_unmasked  =  features_unmasked.cpu().numpy() 
                                        #print (np_features_masked)  #this is the feature we want to compare



                                        # Here We need to compare 'np_features_masked' with 'np_features_unmasked' using  three appracohes
                                        # the Euclidean norm of a vector can be defined as the square root of the inner product of a vector with itself.
                                        np_features_masked_normalized=np_features_masked/np.sqrt(np.dot(np_features_masked,np.transpose(np_features_masked)))
                                        np_features_unmasked_normalized=np_features_unmasked/np.sqrt(np.dot(np_features_unmasked,np.transpose(np_features_unmasked)))


                                        MSE_value=np.sum(np.abs(np_features_masked-np_features_unmasked)) / len(np_features_masked[0])
                                        DOT_value=np.dot(np_features_masked,np.transpose(np_features_unmasked))
                                        DOT_NORM_value= np.dot(np_features_masked_normalized,np.transpose(np_features_unmasked_normalized))
                                        

                                        # if same == true do... if same ==false do this .. use some variables 
                                        # Please note that for the case of MSE threshold the lower the difference between
                                        # the MSE features the closer the images.
                                        # However, for the dot product the higher the product the closer the images. 

                                        if same :
                                            print ("same identity")
                                            similar_mse.append(MSE_value)
                                            similar_dot.append(DOT_value)
                                            similar_ndot.append(DOT_NORM_value)
                                        
                                        else :
                                            print ("different identity")
                                            different_mse.append(MSE_value)
                                            different_dot.append(DOT_value)
                                            different_ndot.append(DOT_NORM_value)
                                             
            
                    print ("similar_mse: ",similar_mse)
                    print ("different_mse: ",different_mse)
                    if np.min(similar_mse)< MSE_treshold and np.min(different_mse)> np.min(similar_mse):
                        correct_MSE+=1

                    if np.max(similar_dot) > DOT_threshold and np.max(different_dot)< np.max(similar_dot):
                        correct_DOT+=1

                    if np.max(similar_ndot) > DOT_NORM_threshold and np.max(different_ndot)< np.max(similar_ndot):
                        correct_DOT_NORM+=1


print("Total running time:", datetime.now() - start)
print ("Accuracy for the MSE approach is {}".format (correct_MSE/count))
print ("Accuracy for the DOT approach is {}".format (correct_DOT/count))
print ("Accuracy for the DOT_NORM approach is {}".format (correct_DOT_NORM/count))


