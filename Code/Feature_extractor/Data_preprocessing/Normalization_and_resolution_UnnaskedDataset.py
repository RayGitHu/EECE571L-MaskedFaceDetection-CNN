
#### This code does normalization and resolution-check for the unmasked dataset used in the training part located at
#### the following path: /Data/Feature_extractor/Unmasked_dataset/


######## Train Dataset Normalization ################


import shutil, random, os
import cv2
import numpy as np

#genders=['males', 'females']
types=['train']
for type in types: 
    directory_path = '/EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Unmasked_dataset/'+type 
    folders= os.listdir(directory_path) # list of all the folder names
    
    count=0
    b_mean=[]
    b_std= []
    g_mean=[]
    g_std= []
    r_mean=[]
    r_std= []
    for folder in folders: # looping through all the gender folders
        if folder not in  ".DS_Store":
                count+=1
                folder_path= directory_path +"/"+ folder
                files= os.listdir(folder_path) # names of the files inside the folder
                
            
                for file in files:
                    if file not in  ".DS_Store":
                        img = cv2.imread(folder_path+"/"+ file)
                        b = img[:,:,0]/255
                        g = img[:,:,1]/255
                        r = img[:,:,2]/255

                        b_mean.append(b.mean())
                        b_std.append(b.std())
                        g_mean.append(g.mean())
                        g_std.append(g.std())
                        r_mean.append(r.mean())
                        r_std.append(r.std())

    B_mean= np.mean (b_mean)
    B_std= np.mean (b_std)
    G_mean=np.mean(g_mean) 
    G_std=np.mean (g_std) 
    R_mean=np.mean(r_mean) 
    R_std=np.mean (r_std) 
    print (" Normalziation for train dataset:")
    print (R_mean,G_mean, B_mean,R_std,G_std, B_std)            
   

# NOTE:
# 
# - mean train dataset: [0.4415734549019608 0.388225627197976 0.3462771010120177]
# 
# - STD train dataset: [0.2711501212166684 0.24437697540137235 0.2350548283233844]



######## Validation Dataset Normalization ################

import shutil, random, os
import cv2
import numpy as np

#genders=['males', 'females']
types=['val']
for type in types: 
    directory_path = '/EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Unmasked_dataset/'+type 
    folders= os.listdir(directory_path) # list of all the folder names
    
    count=0
    b_mean=[]
    b_std= []
    g_mean=[]
    g_std= []
    r_mean=[]
    r_std= []
    for folder in folders: # looping through all the gender folders
        if folder not in  ".DS_Store":
                count+=1
                folder_path= directory_path +"/"+ folder
                files= os.listdir(folder_path) # names of the files inside the folder
                
               
                for file in files:
                    if file not in  ".DS_Store":
                        img = cv2.imread(folder_path+"/"+ file)
                        b = img[:,:,0]/255
                        g = img[:,:,1]/255
                        r = img[:,:,2]/255

                        b_mean.append(b.mean())
                        b_std.append(b.std())
                        g_mean.append(g.mean())
                        g_std.append(g.std())
                        r_mean.append(r.mean())
                        r_std.append(r.std())

    B_mean= np.mean (b_mean)
    B_std= np.mean (b_std)
    G_mean=np.mean(g_mean) 
    G_std=np.mean (g_std) 
    R_mean=np.mean(r_mean) 
    R_std=np.mean (r_std) 
    print (" Normalziation for validation dataset:")
    print (R_mean,G_mean, B_mean,R_std,G_std, B_std)             
                            

# NOTE:
#     
# - mean validation dataset: [0.4337609743200506 0.38305414800759013 0.34232177254901963]
# - STD validation dataset: [0.2697559890251564 0.2456542223409202 0.23599863044380645]





######## Checking train image reoslutions ################


import cv2
import numpy as np

types=['train']
for type in types: 
    directory_path = '/EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Unmasked_dataset/'+type 
    folders= os.listdir(directory_path) # list of all the folder names
    
    count=0
    for folder in folders: # looping through all the gender folders
        if folder not in  ".DS_Store":
               
                folder_path= directory_path +"/"+ folder
                files= os.listdir(folder_path) # names of the files inside the folder
                
                for file in files: 
                    # loading the image
                    img = cv2.imread(folder_path+"/"+ file)

                    # fetching the dimensions
                    wid = img.shape[1]
                    hgt = img.shape[0]
                    
                    # displaying the dimensions
                    print(str(wid) + "x" + str(hgt))
                    
                    if wid == 250 and hgt == 250:
                         count+=1
                        
                   
            
print ("Total numbeer of train images with a 205x250 resolution is {} ".format (count))    



######## Checking validation image reoslutions ################

import cv2
import numpy as np

types=['val']
for type in types: 
    directory_path = '/EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Unmasked_dataset/'+type 
    folders= os.listdir(directory_path) # list of all the folder names
    
    count=0
    for folder in folders: # looping through all the gender folders
        if folder not in  ".DS_Store":
               
                folder_path= directory_path +"/"+ folder
                files= os.listdir(folder_path) # names of the files inside the folder
                
                for file in files: 
                    # loading the image
                    img = cv2.imread(folder_path+"/"+ file)

                    # fetching the dimensions
                    wid = img.shape[1]
                    hgt = img.shape[0]
                    
                    # displaying the dimensions
                    print(str(wid) + "x" + str(hgt))
                    
                    if wid == 250 and hgt == 250:
                         count+=1
                        
                   
            
print ("Total numbeer of validation images with a 205x250 resolution is {} ".format (count))    



