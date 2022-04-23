
#Written by Hamid Reza Tohidypour for UBC's DML and UBC's ECE571
#Confidential do not distribute it beyond your team of ECE571



from __future__ import print_function, division

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
#from pytorch_model_summary import summary
from datetime import datetime
from argparse import ArgumentParser
#plt.ion()   # interactive mode
from torch.utils.data import Dataset
import cv2
import numpy
from PIL import Image
from torch import Tensor
from typing import Tuple, List, Optional
import torchvision.transforms.functional as TF
from pathlib import Path


# Mean and STD of the unmaksed dataset (train dataset) 
meanDataset =[0.4415734549019608, 0.388225627197976, 0.3462771010120177]
stdDataset =[0.2711501212166684, 0.24437697540137235, 0.2350548283233844]


parser = ArgumentParser()
parser.add_argument('--resolution', default=224, type=int, help='Resized to this resoloution')
parser.add_argument('--dataDir', type=str,  default = 'dataset', help='Location of the dataset including train and val folders ')
parser.add_argument('--data2ndDirTrain', type=str,  default = 'dataset probably not needed', help='Location of the dataset including train and val folders ')

parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')

parser.add_argument('--numclasses', default=1000, type=int,
                    help='Number of Quality levels')
parser.add_argument('--LocationofSavedModel', type=str,   help='Location of a saved model')

parser.add_argument('--cropXSize', default=224, type=int, help='The crop size')
parser.add_argument('--cropYSize', default=224, type=int, help='The crop size')
parser.add_argument('--enableHFlip', default=0, type=int,
                    help='enable HorizentalFlip for train and validation')

parser.add_argument('--trainClassesFile', default="trainClasses.txt", type=str,
                    help='UseDataParallel')


parser.add_argument('--Classesnames', default="synset_words.txt", type=str,
                    help='UseDataParallel')


parser.add_argument('--NameOfResultsFile', default="testresult", type=str,
                    help='name of the result file')



parser.add_argument('--valOrtest', type=str,  default = 'test', help='validation or test')

args = parser.parse_args()






def ResizedTensor(image, size):

    dim=size
    resized = cv2.resize(image, dim, interpolation =  cv2.INTER_LINEAR )
    #print(resized.shape)
    return resized

def CenterCropTensor(image, size):

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




def RandomResizedCropTensor(image, size):

    scale=(0.08, 1.0)
    ratio=(3. / 4., 4. / 3.)
    interpolation=Image.BILINEAR

    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
          ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.
        """


        #print( "##################",image.shape) #image.size())
        width, height, channel = image.shape# image.size() #TF._get_image_size(image)
        area = height * width

        for _ in range(10):
          target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
          log_ratio = torch.log(torch.tensor(ratio))
          aspect_ratio = torch.exp(
             torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
          ).item()

          w = int(round(math.sqrt(target_area * aspect_ratio)))
          h = int(round(math.sqrt(target_area / aspect_ratio)))

          if 0 < w <= width and 0 < h <= height:
             i = torch.randint(0, height - h + 1, size=(1,)).item()
             j = torch.randint(0, width - w + 1, size=(1,)).item()
             return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    i, j, h, w = get_params(image, scale, ratio)
    if (h%2==1):
       h=h-1
    if (w%2==1):
       w=w-1
    #print("#############",i,",", j, ",",h,",", w )
    #return TF.resized_crop(image, i, j, h, w, size, interpolation)
    img=image[j:j+w,i:i+h,:];
    dim=size
    #print(img.shape)
    #print(dim)
    #print(img)
    resized = cv2.resize(img, dim, interpolation =  cv2.INTER_LINEAR )
    return resized




class RandomResizedCrop(torch.nn.Module):
    """Crop the given image to random size and aspect ratio.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
        interpolation (int): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        return resized # F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class Datasets(Dataset):
     #def __init__(self, dataDir,phase):
     def __init__(self, cropXsize,cropYsize,dataDir,data2ndDirTrain,enableHFlip,phase, numClasses, fileLables):
         self.cropXsize=cropXsize
         self.cropYsize=cropYsize
         self.phase=phase
         self.enableHFlip=enableHFlip
         self.fileLables=fileLables
         num_train_classes=0
         print("dataDir",dataDir)
         train_image_dir=os.path.join(dataDir,phase)
         print("dataDir",train_image_dir)
         for trainSubFolders in os.listdir(train_image_dir): #for HDRSDR in os.listdir(dir_path):
             if os.path.isfile(os.path.join(train_image_dir,trainSubFolders)):
                continue
             train_image_classes=os.path.join(train_image_dir,trainSubFolders)
             num_train_classes+=1
         #for trainSubFolders in os.listdir(train_image_classes):


         if (phase=='train' and data2ndDirTrain !=None):
            train_image_dir=os.path.join(data2ndDirTrain,phase)
            for trainSubFolders in os.listdir(train_image_dir): #for HDRSDR in os.listdir(dir_path):
               if os.path.isfile(os.path.join(train_image_dir,trainSubFolders)):
                  continue
               train_image_classes=os.path.join(train_image_dir,trainSubFolders)
               num_train_classes+=1




         self.files=[]
         self.classes=[]
         self.classesalreadyreadfromfile=[]
         self.labels=[]

         self.files_classes=[]
         self.labels_classes=[]


         for i in range (0,numClasses): # num_train_classes):
             self.files_classes.append([])
             self.labels_classes.append([])
         #print(train_files)

         i=0




         self.classes=copy.copy(self.fileLables)
         train_image_dir=os.path.join(dataDir,phase)
         for trainSubFolders in os.listdir(train_image_dir): #for HDRSDR in os.listdir(dir_path):
             #print("$$$$$$$$$$",trainSubFolders)
             if os.path.isfile(os.path.join(train_image_dir,trainSubFolders)):
                continue
             train_image_classes=os.path.join(train_image_dir,trainSubFolders)
             #self.classes.append(trainSubFolders)
             self.classesalreadyreadfromfile.append(trainSubFolders)
             #print(trainSubFolders)
             
             if phase=='val' or phase!='test':
                i=self.fileLables.index(trainSubFolders)
             for classimages in sorted(os.listdir(train_image_classes)):

                 if os.path.isdir(os.path.join(train_image_dir,trainSubFolders,classimages)):
                   continue
                 #image = cv2.imread(os.path.join(train_image_classes,classimages), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
                 #if (len(image.shape)<3):
                 #   continue;
                 self.files_classes[i].append(os.path.join(train_image_classes,classimages))
                 #print(os.path.join(train_image_classes,classimages))
                 self.labels_classes[i].append(i) #trainSubFolders)
                 #if ("Very_annoying" in trainSubFolders):
                 #    self.files_classes[i].append(os.path.join(train_image_classes,classimages))
                 #    self.labels_classes[i].append(i) #trainSubFolders)
                 #print(os.path.join(train_image_classes,classimages))
                  
             i=i+1

         #print ("order ",self.files_classes[0])

         if (phase=='train' and data2ndDirTrain !=None):
            train_image_dir=os.path.join(data2ndDirTrain,phase)
            for trainSubFolders in os.listdir(train_image_dir): #for HDRSDR in os.listdir(dir_path):
                if os.path.isfile(os.path.join(train_image_dir,trainSubFolders)):
                   continue
                train_image_classes=os.path.join(train_image_dir,trainSubFolders)

                if (trainSubFolders in self.classesalreadyreadfromfile):
                   print ("Error: the class repeated in the second train folder ", trainSubFolders )
                   exit;
                #self.classes.append(trainSubFolders)
                #print(trainSubFolders)
                i=self.fileLables.index(trainSubFolders)
                for classimages in os.listdir(train_image_classes):
                   self.files_classes[i].append(os.path.join(train_image_classes,classimages))
                   #print(os.path.join(train_image_classes,classimages))
                   self.labels_classes[i].append(i) #trainSubFolders)
                   #if ("Very_annoying" in trainSubFolders):
                   #    self.files_classes[i].append(os.path.join(train_image_classes,classimages))
                   #    self.labels_classes[i].append(i) #trainSubFolders)
                  
                i=i+1



         #if not "Very_annoying" in self.classes:
         #   print ("Error: the name of the folder for the frames with the lowest quality should be Very_annoying, please change fix it and run the code again")
         #   exit();



         if 2==1: #phase =='test':
           self.files= copy.copy(self.files_classes[0])
           self.labels=copy.copy(self.labels_classes[0])
           print("##########")
           print(self.files_classes) #[0])
           print("$$$$$$$$$$$")
           print(self.labels_classes)

         else:
           q=0
           copy_of_files_classes = copy.copy(self.files_classes)
           copy_of_labels_classes = copy.copy(self.labels_classes)
           while (self.files_classes): # (len(self.SeqNameHas)>0): #(self.SeqNameHas!=None):
             #print(self.SavedSeqs) #[i], len(self.SavedSeqs))
             self.files_classes= copy.copy(copy_of_files_classes)
             self.labels_classes= copy.copy(copy_of_labels_classes)

             for i in range (0, len(self.files_classes)):
               #print(i)
               if (len(self.files_classes[i])>0):
                  #print(self.SeqNameHas[i])
                  q=q+1
                  self.files.append(self.files_classes[i].pop())
                  self.labels.append(self.labels_classes[i].pop())
               else:
                  #print(self.files_classes)
                  #print("remove",i, len(self.files_classes))
                  #print("remove",i, len(self.files_classes)," ",self.files_classes[i])
                  if (i==len(self.files_classes)-1):
                    copy_of_files_classes.remove(self.files_classes[i]) #=[]
                    copy_of_labels_classes.remove(self.labels_classes[i]) #=[]
                    #print("######")
                    break;
                  else:
                    copy_of_files_classes.remove(self.files_classes[i]) 
                    copy_of_labels_classes.remove(self.labels_classes[i]) 

          #print("#########",self.files_classes)
          #if (phase=='train'):
          #    f=open("trainClasses.txt", 'w') 
          #    for i in range (0,len(self.classes)):
          #       f.write(self.classes[i])
          #       f.write('\n')
          #    f.close() 













     def __getitem__(self, item):
         #print(item, "item")
         image = cv2.imread(self.files[item], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
         #print(self.files[item],image.shape )

         channels=image .shape;

         if (len(image.shape)<3):
              channels=2;
         else:
           channels=channels[2];
         #print("image .shape",self.files[item]," ",image .shape," ",channels)

         
         #print("image .shape",image .shape[:2])
         if (len(image.shape)<3):
            output = [[], [], self.files[item],2]
            #print (self.files[item], image.shape)
            return output             
         else:
            try:
               channel,height_img, width_img = image.shape
            except AttributeError:
               print("shape not found for ", self.files[item])

         Rangeoffset_h=height_img-self.cropXsize
         Rangeoffset_w=width_img-self.cropYsize
         #offset_h = torch.randint(low=0,high=Rangeoffset_h,size=(1,))
         #offset_w = torch.randint(low=0,high=Rangeoffset_w,size=(1,))
         #image1=image[:,offset_h:offset_h+self.cropXsize,offset_w:offset_w+self.cropYsize]
         #print("image .shape1 ",image .shape)
         image1=image



         image1=image1/255;
         for ch in range (0,3):
           
            image1[:,:,ch]=(image1[:,:,ch]-meanDataset[ch])/stdDataset[ch]
         #image1=(image1-0.5)*2
         #high_resolution = high_resolution*2.0 - 1.0;

         #if random() > 0.5:
         #    high_resolution = TF.vflip(high_resolution)
         #    low_resolution = TF.vflip(low_resolution)




         #image1=RandomResizedCropTensor(Image.fromarray(image1), (224,224))
         #image1= np.asarray(image1)

         if self.phase == 'train':
            image1=RandomResizedCropTensor(image1, (224,224))
         else:
            image1=ResizedTensor(image1,(256,256))
            image1=CenterCropTensor(image1,(224,224))

         if (image1 .shape[2]==3):
             image1 = numpy.rollaxis(image1,2,0)


         image1 = torch.from_numpy((image1)) # TF.to_tensor(image1) #torch.from_numpy(image1) #TF.to_tensor(image)

         if  self.enableHFlip and self.phase == 'train' and random() > 0.5:
             #image1 = TF.hflip(image1)
             #image1 = cv2.flip(image1, 1)
             image1 = torch.flip(image1, [2]) 


         #print("image .shape11 ",image1 .shape)
         #label= torch.tensor(self.classes[item])
         label= torch.from_numpy(numpy.array(self.labels[item]))
         #print("label",label)
         #output = {'inputs': image1, 'labels': label} #self.classes[item] }
         output = [image1, label, self.files[item],channels]
         #print (output)
         return output


     def __len__(self):
        return len(self.files) #image_file_name)




def main():




  print(args)

  print("meanDataset:",meanDataset)
  print("stdDataset: ",stdDataset)

  fileLables=[]
    
  with open(args.trainClassesFile, 'r') as f:
      while True:
        a1 = f.readline()
        if not a1:
           break
        b=a1.strip()#.split('\n')
        fileLables.append(b)
        #print("val ",len(fileLables))
      #print("val ", fileLables)


  #print(fileLables,"%%%%%%%%%%%%")
  nameLables=[]
    
  with open(args.Classesnames, 'r') as f:
      while True:
        a1 = f.readline()
        if not a1:
           break
        b=a1.strip()#.split('\n')
        nameLables.append(b)
        #print("val ",len(fileLables))
      #print("val ", fileLables)



  data_dir={}
  data_dir[args.valOrtest] = args.dataDir


  # Data augmentation and normalization for training
  # Just normalization for validation
  #data_transforms = {
  #        'test': transforms.Compose([
  #         transforms.RandomCrop(args.resolution),
  #         #transforms.RandomHorizontalFlip(),
  #         transforms.ToTensor(),
  #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  #     ])

  #}


  #image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
  #                                       data_transforms[x])
  #                  for x in ['test']}

  #image_datasets = {x: Datasets( data_dir,x)
  #                   for x in ['val']}


  #dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                 #                            shuffle=True, num_workers=args.num_workers) #num_workers was four
                 #for x in ['val']}

  image_datasets = {x:Datasets(args.cropXSize, args.cropYSize, data_dir[x],args.data2ndDirTrain,args.enableHFlip,x, args.numclasses, fileLables)
                    for x in [args.valOrtest]}  #'val' args.valOrtest


  #print("image_datasets[x]",image_datasets)
  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=1,
                                             shuffle=False, num_workers=args.num_workers)
                for x in [args.valOrtest]}

  dataset_sizes = {x: len(image_datasets[x]) for x in [args.valOrtest]}
  class_names = image_datasets[args.valOrtest].classes

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  classifier = torch.load(args.LocationofSavedModel).eval()
  classifier.to(device)

  #print(image_datasets[args.valOrtest].files)
  print("Total validation files",len(image_datasets[args.valOrtest].files))

  print("number of val classes ", len(image_datasets[args.valOrtest].classes))

  #path = os.path.join(os.path.dirname(__file__), 'Dataset/Test')
  #test_dataset = notMNIST(path)
  #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
  #classifier = torch.load('models/{}.pt'.format(MODEL_NAME)).eval()
  correct = 0 # torch.tensor(0,dtype=torch.float32)
  correct_top5 = 0

  confusion_matrix = torch.zeros(args.numclasses,2)

  #for _, data in enumerate(dataloaders , 0):
  #	test_x, test_y = data




  num_of_colour_images=0;
  for inputs, labels, files, channels in dataloaders['test']: #[phase]:


        #print("files ",files)
        if channels ==3:
          num_of_colour_images=num_of_colour_images+1
          inputs = inputs.to(device, dtype=torch.float)
          labels = labels.to(device, dtype=torch.long)
          #print(labels.shape )
          outputs = classifier(inputs)
          #print (outputs)
          _, preds = torch.max(outputs, 1)
          _, preds5top = torch.topk(outputs, 5)
          #if preds == labels.data: #y_hat == labels:
          #   correct += 1
          correct+=torch.sum(preds == labels.data)

          #print("labels.data",labels.data)
          confusion_matrix[labels.data,1]+=1
          if preds == labels.data:
             confusion_matrix[labels.data,0]+=1
          #else:
          #Path(files).name
          #print("files",files[0]);


          res=preds5top.cpu().numpy()
          for i in range (0,5):
            correct_top5+=torch.sum(res.item(i) == labels.data)


     
 

  #print("Total Accuracy={}".format(correct.double() / dataset_sizes[args.valOrtest]))
  #print("Total Accuracy-top5={}".format(correct_top5.double() / dataset_sizes[args.valOrtest]))
  print("Total Accuracy={}".format(correct.double() / num_of_colour_images))
  print("Total Accuracy-top5={}".format(correct_top5.double() / num_of_colour_images))

  print("#########################################################")
  print("classes and their accuracies:")
  for i in range(0,len(image_datasets[args.valOrtest].classes)):
    print(class_names[i]," ", format(confusion_matrix[i,0].double() / confusion_matrix[i,1].double()))

  print("Accuracy of every quality level {}".format(confusion_matrix[:,0].double() / confusion_matrix[:,1].double()))



if __name__ == '__main__':
    main()