# License: BSD
#Written by by Hamid Reza Tohidypour for DML's UBC and ECE 571L
#Confidential do not distribute it beyond your team 
#some parts of the training part borrowed from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
#from pytorch_model_summary import summary
from datetime import datetime
from argparse import ArgumentParser
#plt.ion()   # interactive mode

    


parser = ArgumentParser()
parser.add_argument('--cropSize', default=224, type=int, help='The crop size')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--dataDir', type=str,  default = 'dataset', help='Location of the dataset including train and val folders ')
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs for training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')

parser.add_argument('--num_workers', default=1, type=int, help='Number of workers')
parser.add_argument('--DecayLRafterThisNumEpoch', default=1000, type=int, help='Decay learning rate after this number of epochs')
parser.add_argument('--weightDecay', default=0.5, type=float, help='Weight Decay')

parser.add_argument('--save_iter', default=50, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')
parser.add_argument('--numclasses', default=100, type=int,
                    help='Number of classes')
parser.add_argument('--logDir', type=str, default='log', help='Path to th log directory.')
parser.add_argument('--DirTosaveModel', type=str,  default='savedModelLogs', help='Path to the save the model and the logs.')
parser.add_argument('--LocationofSavedModel', type=str,   help='Location of a saved model if you want to continue training')



def train_model(model, criterion, optimizer, scheduler,dataloaders , dataset_sizes, device, args, class_names, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    writer = SummaryWriter(log_dir=args.DirTosaveModel+'/logs')
    confusion_matrix = torch.zeros(5,5)
    itr=0
    justSavedLog =0 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                itr=itr+1
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    pr = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if (itr%args.save_iter==0):
                            #print(loss.item())
                            corr=torch.sum(preds == labels.data)
                            corr = corr.double()/inputs.size(0)
                            writer.add_scalar('Loss/train vs #iteration', loss.item()*1.0, itr)
                            writer.add_scalar('Accuracy/train vs #iteration', corr, itr)
                            tm = datetime.now().strftime("%Y%m%d%H%M%S")
                            torch.save(model,args.DirTosaveModel+ '/models/' + tm + "_" + str(itr) + '_model.h5')
                            justSavedLog=1
                            inp=255* (inputs+1.0)/2.0
                            inp_= torchvision.utils.make_grid(inp)
                            inp_=torch.tensor(inp_,dtype=torch.uint8)
                            writer.add_image('Training images', inp_, itr)

            
                if (justSavedLog==1 and phase == 'val'):
                    #print(loss.item())
                    corr=torch.sum(preds == labels.data)
                    corr = corr.double()/inputs.size(0)
                    writer.add_scalar('Loss/validation vs #iteration', loss.item()*1.0, itr)
                    writer.add_scalar('Accuracy/validation vs #iteration', corr, itr)
                    justSavedLog=0

                    inp=255* (inputs+1.0)/2.0
               
                    inp_= torchvision.utils.make_grid(inp)
                    inp_=torch.tensor(inp_,dtype=torch.uint8)
                    writer.add_image('Validation images', inp_, itr)



                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


             



            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model,args.DirTosaveModel+ '/models/' +'modelWithBestValidationAcc.h5')

        print()
    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def main():

   args = parser.parse_args()



   if not os.path.exists(args.DirTosaveModel):
       os.makedirs(args.DirTosaveModel)
   if not os.path.exists(args.DirTosaveModel+'/models'):
       os.makedirs(args.DirTosaveModel+'/models')


   # Data augmentation and normalization for training
   # Just normalization for validation
   data_transforms = {
          'train': transforms.Compose([
           #Crop a random portion of image and resize it to a given size.
           transforms.RandomResizedCrop(224),
           #transforms.Resize(224,224),#.(224),
           #transforms.RandomCrop(args.cropSize),
           #Horizontally flip the given image randomly with a given probability.
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize( [0.4415734549019608, 0.388225627197976, 0.3462771010120177]  ,[0.2711501212166684, 0.24437697540137235, 0.2350548283233844]  )   #Modify this part and use the mean and SD you found for the train dataset
       ]),
       'val': transforms.Compose([
           #transforms.RandomResizedCrop(224),
           #transforms.Resize(224, 224),
           #transforms.Resize(512),
           #transforms.RandomCrop(args.cropSize),
           #Crops the given image at the center.
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize( [0.4337609743200506, 0.38305414800759013, 0.34232177254901963] , [0.2697559890251564, 0.2456542223409202, 0.23599863044380645]  )   #Modify this part and use the mean and SD you found for the val dataset
       ]),
   }

   data_dir = args.dataDir 
   image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                     for x in ['train', 'val']}
   dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers) #num_workers was four
                 for x in ['train', 'val']}
   dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
   class_names = image_datasets['train'].classes

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




   if args.LocationofSavedModel != None:
        print("Loading pre-trained model...")
        model_ft = torch.load(args.LocationofSavedModel)

        if 'Org' in args.LocationofSavedModel or 'org' in args.LocationofSavedModel:
          if 'Vgg' in args.LocationofSavedModel or 'vgg' in args.LocationofSavedModel:
            num_ftrs = model_ft.classifier[3].out_features
            model_ft.classifier[6]= model_ft.fc = nn.Linear(num_ftrs, args.numclasses)
        print(model_ft)
    
    
       
    
   model_ft = model_ft.to(device) #model.to(device) 


   criterion = nn.CrossEntropyLoss()

   # Observe that all parameters are being optimized
   optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr)

   # Decay LR by a factor of gamma after certain number of epochs
   #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

   exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.DecayLRafterThisNumEpoch, gamma=args.weightDecay)

   model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device, args, class_names,
                          num_epochs=args.epochs )
   


if __name__ == '__main__':
        main()