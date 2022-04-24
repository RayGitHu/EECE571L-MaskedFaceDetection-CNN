# EECE571L-MaskedFaceDetection-CNN
## Team members
Mobina Mobaraki, Mohamed Zidan, Haoxiang Lei, Rui Zhong

## GitHub repositories used in this project
Face Recognition Datasets:https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

To put mask on the faces: https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/face_mask_adding/FMA-3D

To train the vgg-19 model: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## Requirements
### Programming language 
  Python 3.7
### Deep Learning Frameworks
  Pytorch 1.11.0
### Operating System
  Mac 12.1/ Windows10/ Linux (Compute Canada)
### Requirements and Dependenies
  All the requirement/dependencies with their versions can be found in Dependencies/ env_face_dependencies.txt
### Deep neural network
  VGG-19 model with 5 convolutional layers, 5 max pooling layers, and 3 fully connected layer. The activation function is ReLU. There is no Softmax layer as Pytorch internally computes the probability of each class as the output of the model.

![image](https://user-images.githubusercontent.com/53150477/164339904-031627ee-86b3-4cb6-bd39-977b88530681.png)

### Virtual environment [ to be updated ]
  Here are steps how we create virtual enviroment:
  
  Step 1: Create a virtual environment named ENV 

     [name@server ~] virtualenv --no-download ~/ENV

  Step 2: Once the virtual environment has been created, it must be activated: 

     [name@server ~] source ~/ENV/bin/activate

  Step 3: Upgrade pip in the environment

    (ENV)[name@server ~]$ pip install --no-index --upgrade pip

  Step 4: Install numpy in the environment

    (ENV)[name@server ~] pip install numpy --no-index

  Step 5: Type the following commands:

    (ENV) [name@server ~]                 
    (ENV) [name@server ~] pip install --no-index torch torchvision torchtext                
    (ENV) [name@server ~] pip install matplotlib                
    (ENV) [name@server ~] pip install torchsummary                
    (ENV) [name@server ~] pip install tensorboard
    
### External Computing Services
  We used Graham Cluster (graham.computecanada.ca) of the Compute Canada which is based on Linux. We installed "WinSCP" to transfer the folders into Compute Canada and "Putty" to run the Linux codes.
  
  Here are steps to run the code on the Graham cluster of the ComputeCanada:
  
  Step 1: Load python 3.7
  
    [name@server ~] module load python/3.7
    
  Step 2: Load other dependencies:
    
    [name@server ~] module load nixpkgs/16.09  gcc/7.3.0 opencv/4.2

  Step 3: Run the test code using the following commands
    
    (ENV) [name@server ~] cd /home/yourusername/scratch
    (ENV) [name@server ~] sbatch test.sh
    
  Step 4: Check to see if the code is running:

    (ENV) [name@server ~] squeue -u yourusername
    
  Step 5: Once it finished running the code. Check the scratch folder and look for the slurm part.





## Dataset
   In our project, we have two major types of datasets, Feature Extractor and Similarity Analysis.

### Feature Extractor
   Feature extractor dataset has 62 identities which includes 50 male and 12 female. It has 1,240 unmasked images and 248 masked images. Each identity has 16 images for training, 4 images for validation and 4 images for test. 
   
   Train and validation images are under path _EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Unmasked_dataset/_. Test images are under path _EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Test_dataset/_.
   
   Feature extractor part has the other dataset named mixed datset which includes 62 indentities and each identity has 16 masked images and 16 unmasked images for training and 4 masked images and 4 unmasked images for validation.
   
   Mixed dataset is under path _EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Mixed_dataset/_.

### Similarity Analysis
   In Similarity Analysis part, we have two datasets, full dataset and removed dataset. Full dataset has 532 images and 38 identities and each identity has 7 masked images and 7 unmasked images. In removed dataset, it still has 38 identities and 266 images. Each identity keeps two masked images and we removed their unmasked version from unmasked directory so that each identity has 2 masked images and 5 unmasked images.
   
   This dataset is under path _EECE571L-MaskedFaceDetection-CNN/Data/Similarity Analysis/_.
   
### Processing
   Before we use this dataset to train our neural network, we should pre-process dataset.
  
#### Nomalization
   We need to find the mean and standard deviation of the dataset. We divided each channel of the images by 255, because they are RGB images and in 8 bits format with the max of 255. Then we found the mean and SD of each channel of the image. This means we have three mean values and three SD values Then we average them across the images. 

#### Augmentation
   To obtain the fixed-size 224×224 ConvNet input images, they were randomly cropped from rescaled training images (one crop per image per SGD iteration). To further augment the training set, the crops underwent random horizontal flipping. The validation dataset was centred cropped. This process is used to avoid overfitting.

#### Put mask on the faces
   We modify the provided face-masking code in this link _https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/face_mask_adding/FMA-3D_ 

For face masking, the code does the follwoing:
  - Face detected using 4 values.
  -  Face aligned using 106 extracted landmarks values
  -   Mask applied to on the face.

For more details on face masking process, please see feature extractor sub-section under the Model section below.

## Model 
#### Feature Extractor
The related codes are in Code/Feature_extractor folder. These codes do the following:
  - Normalization and resolution-check for the Unmasked dataset used in the training part*. 
  - Training VGG-19 model on the unmasked train dataset*.
  - Face detection,alignment, and masking for unmasked test dataset (i.e. Data/Feature_extractor/Test_dataset/Test0/test)*.
  - Testing the trained VGG-19 model unmasked test dataset*. 



*For unmasked dataset normalization and resolution-check,the following actions are taken:
  - Each of the train and validation datasets are normalized using their computed mean and STD values.
  - Each of the train and validation datasets are checked to ensure that they have a resolution of 250 x 250.
  
To run the normalization and resolution-check code on your local computer, please do the following :...




*For training the VGG-19 model,the following actions are taken:
  - Weight initialisation: We fine-tuned the VGG-19 convolutional network. To be more specific,  we used the pre-trained VGG-19 model on ImageNet dataset with 1000 categories as an initialization for the task of recognizing masked faces. Rest of the training looks as usual ( forward and backward propagation ).
  
  - Optimization objective: The training is carried out by optimising the cross-entropy loss function using mini-batch Adam algorithm.The batch size is set to 32, and all the parameters are being optimized here.

  - Learning rate: The learning rate was set to 1e-4. The learning was stopped after 100 epochs for the model trained on unmasked dataset and after 250 epochs for the model trained on the mixed dataset.

  - Regularization: The learning rate was set to decay by a factor of 0.5 after 1000 epochs and training was also regularized by dropout regularization for the first two-fully connected layers where the dropout ratio was set to 0.5.


  - Input images pre-processing & augmentation: To obtain the fixed-size 224×224 ConvNet input images, they were randomly cropped from rescaled training images. To further augment the  training set, the crops underwent random horizontal flipping. Also, the training dataset was normalized using its computed mean and STD.For the validation dataset was centred cropped. Also, the validation dataset was normalized using its computed mean and STD.
  
To run the train code on Compute Canada, please do the following :...




*For face detection, alignment,and masking,the following actions are taken:
  - Face detected using 4 values.
  - Face aligned using 106 extracted landmarks values
  - Maksed applied to on the face.
  - Four codes were modified.
     - _face_detect.py_ located at /Code/Feature_extractor/Face_masking/FaceX-Zoo/face_sdk/api_usage/face_detect.py 
     - _face_alignment.py_ located at /Code/Feature_extractor/Face_masking/FaceX-Zoo/face_sdk/api_usage/face_alignment.py 
     - _add_mask_one.py_ located at /Code/Feature_extractor/Face_masking/FaceX-Zoo/addition_module/face_mask_adding/FMA-3D/add_mask_one.py
     - _render.pyx_ located at  /Code/Feature_extractor/Face_masking/FaceX-Zoo/addition_module/face_mask_adding/FMA-3D/utils/cython/render.pyx  
 
For _face_detect.py_,_face_alignment.py_ , and _add_mask_one.py_  you can open each code file and see how many code lines were. For _render.pyx_ , we had to modify _line #57_ by replacing _numpy.empty()_ with _numpy.zeros()_. This small change in line #57 helped us to stop getting the noisy masked images. Please note that  _numpy.empty()_  is a "speed up" method while _numpy.zeros() is a  "no speed up" method.

To run the face_detection_alignment_masking code on Compute Canada, please do the following :...




*For testing the VGG-19 model,the following actions are taken:
- Two datasets of unseen idenitiy images, one unmaksed and other masked, are fed to the test code.
- The code computes the accuracy per identity class and the overall accuracy. 

To run the test code on Compute Canada, please do the following :...



#### Similarity Analysis
The related codes are in the Code/Similarity_analysis folder. These codes get the two images and apply the feature extractor on them to calculate the feature vector. 

In the threshold code, three similarity functions (Mean Absolute Error, Dot product, and Normalized Dot product) are applied on the feature vectors and are averaged to find a threshold value.

To run the code on Compute Canada, please do the following:

    -Run the batch file to generate an Excel file with the MAE,DOT, and DOT_NORM threshold values for the 38 identities.
	        	cd ~/scratch && sbatch Threshold_mae_dot_ndot_Full_dataset_Without_FC3_github.sh
            
    
    - Using Excel Avg function, take the average threshold value for each column of the MAE,DOT, and DOT_NORM approaches. You should end up with three threshold values. For your connivence, the final three threshold values were added to the “Accuracy_mae_dot_ndot_Full_dataset_Without_FC3.py” code file

Note: 

		- If you want to compute  MAE/ Dot product/ normalized dot thresholds for removed dataset (Without FC3 layer), please do the following:

			- Rename "Threshold_mae_dot_ndot_Full_dataset_Without_FC3.sh" to "Threshold_mae_dot_ndot_Removed_dataset_Without_FC3.sh"

			- Rename  "Threshold_mae_dot_ndot_Full_dataset_Without_FC3.py" to "Threshold_mae_dot_ndot_Removed_dataset_Without_FC3.py"

			- Make sure that you replace line 26 inside "Threshold_mae_dot_ndot_Removed_dataset_Without_FC3.sh" with  the following :

  				 python -u /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Code/Similarity_analysis/Threshold/Threshold_mae_dot_ndot_Removed_dataset_Without_FC3.py  --dataset_path /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Data/Similarity_analysis/Removed_dataset --LocationofSavedModel /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Model/modelWithBestValidationAcc_unmasked_masked_250epochs_v4.h5
 

			- After deciding whether or not to use the FC3 layer of the VGG-19 feature extractor model (Lines 86-105), run the batch file to generate an Excel file containing the threshold values for each of the 38 identities.

				cd ~/scratch && sbatch Threshold_mae_dot_ndot_Removed_dataset_Without_FC3.sh

			-  Using Excel Avg function, take the average threshold value for each column of the MAE,DOT, and DOT_NORM approaches. You should end up with three threshold values. For your connivence, the final three threshold values were added to the “Accuracy_mae_dot_ndot_Full_dataset_Without_FC3.py” code file.



In the accuracy code, two new feature vectors are compared with the threshold value and are judged as "same" or "different" identity. Based on the two main conditions which are elaborated in our report, the accuracy of the prediction is evaluated. 

To run the code on Compute Canada, please do the following:

	-Run the batch file for computing accuracy score values for the three approaches 
		cd ~/scratch && sbatch Accuracy_mae_dot_ndot_Full_dataset_Without_FC3_github.sh



Note:

	If you want to compute accuracy scores for removed dataset (Without FC3 layer, please do the following:

		- Rename ""Accuracy_mae_dot_ndot_Full_dataset_Without_FC3.sh" to "Accuracy_mae_dot_ndot_Removed_dataset_Without_FC3.sh"

		- Rename ""Accuracy_mae_dot_ndot_Full_dataset_Without_FC3.py" to "Accuracy_mae_dot_ndot_Removed_dataset_Without_FC3.py"

		-  Make sure that you replace line 26 inside "Accuracy_mae_dot_ndot_Removed_dataset_Without_FC3.sh" with 

			python -u /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Code/Similarity_analysis/Accuracy/Accuracy_mae_dot_ndot_Removed_dataset_Without_FC3.py  --dataset_path /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Data/Similarity_analysis/Removed_dataset --LocationofSavedModel /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Model/modelWithBestValidationAcc_unmasked_masked_250epochs_v4.h5
 

		- After deciding whether or not to use the FC3 layer of the VGG-19 feature extractor model, run the batch file to compute the accuracy scores for each of the MAE,DOT, and DOR_NORM approaches:

			cd ~/scratch && sbatch Accuracy_mae_dot_ndot_Removed_dataset_Without_FC3.sh




In the weighting_model code, the error between the two feature vectors is calculated and a classification model is trained to apply optimal weights on each entity of the feature vector. This classification model determines if the two masked and unmasked images corresponds to the same/ different people based on the error between their feature vectors. 

To run the code please change the following variables:
EPOCHS = The number of epochs for training the model (default=20)
LEARNING_RATE = The learning rate for training the model( default=0.001)
BATCH_SIZE = The batch size for training the model( default=64)
model_dir= a directory to the feature extractor model (you can refer to 'model/modelWithBestValidationAcc_unmasked_100epochs_v2.h5')
root_unmasked=a directory to the unmasked images (Please note that all the unmaksed images should be in one folder)
root_masked=a directory to the masked images that you want to identify (Please note that all the maksed images should be in one folder)


## DEMO
This folder includes a demo code of our project. The app requires a directory of the masked images and a directory of the unmasked images (The unmasked image of the masked person should be included in the unmasked folder otherwise the app outputs nothing). The app finally outputs the name of the masked people in the masked folder.

To run the code please change the following variables:
model_dir= a directory to the feature extractor model (you can refer to 'model/modelWithBestValidationAcc_unmasked_100epochs_v2.h5')
root_unmasked=a directory to the unmasked images (Please note that all the unmaksed images should be in one folder)
root_masked=a directory to the masked images that you want to identify (Please note that all the maksed images should be in one folder)
classifier_path=a directory to the weighting model. You can refer to 'model/classification_model.h5'

 ## Code WalkThrough
   Please check out the "Code_WalkThrough.ipynb" for detailed instructions.

## Future work
1) The Demo code is based on the weighting model and can only identify the masked images if the identity of the person is included in the training dataset. To identify masked face of people who are not included in the training dataset, as future work the weighting model can be re-trained on triplet loss instead of cross entropy loss also the feature extractor can be re-trained on more masked/unmaksed images with different background and facial appearances.
2) The Demo code cannot identify the person with masked image if the pose of the persn is completely different or if the lightening of the masked image completely different from the unmasked images (different mean and std results in different feature vectors). Re-training the feature extractor and weighting model on bigger and more representative datasets (with various lightening) would expand the usibility of our app.
3) Instead of training the feature extractor on both masked and unmasked images, if we train it on only masked images, the feature extractor may more accurately extract the features from non-occluded areas and the app may identify the masked person more accurately.




   
  
