"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 

Modified by EECE 571L Team 2-Face1-CNN-2021W2 term 
Lines 55-67 & 69-70 & 79-80  ADDED  
"""
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
import os # ADDED
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

if __name__ == '__main__':
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]

    logger.info('Start to load the face landmark model...')
    # load model
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face landmark model!')

    faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

    
    #55-67 & 69-70 & 79-80  ADDED  
    directory_path = '/EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Test_dataset/Test0/test'
    folders= os.listdir(directory_path) # list of all the folder names
    
    for folder in folders: # looping through all the folders
        if folder not in  ".DS_Store":
            folder_path= directory_path +"/"+ folder
            files= os.listdir(folder_path) # names of the files inside the folder
            files_count= len(files)      # number of the files inside the folder 

            to_directory= '/EECE571L-MaskedFaceDetection-CNN/Code/Feature_extractor/Face_masking/FaceX-Zoo/face_sdk/api_usage/temp/face_alignment/'+folder
            #
            os.mkdir(to_directory)
            read_path_txt= '/EECE571L-MaskedFaceDetection-CNN/Code/Feature_extractor/Face_masking/FaceX-Zoo/face_sdk/api_usage/temp/face_detect/'+folder
            for file_ in files:
                if file_ not in  ".DS_Store": 
                    # read image
                    image_path = folder_path  +"/"+ file_
                    image_det_txt_path = read_path_txt +"/"+  file_  +'_detect_res.txt'
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    with open(image_det_txt_path, 'r') as f:
                        lines = f.readlines()
                    try:
                        for i, line in enumerate(lines):
                            line = line.strip().split()
                            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
                            landmarks = faceAlignModelHandler.inference_on_image(image, det)
                            save_path_img = to_directory +"/"+  file_  +  '_' + 'landmark_res' + str(i) + '.jpg'
                            save_path_txt = to_directory +"/"+  file_  + '_' + 'landmark_res' + str(i) + '.txt'
                            image_show = image.copy()
                            with open(save_path_txt, "w") as fd:
                                for (x, y) in landmarks.astype(np.int32):
                                    cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
                                    line = str(x) + ' ' + str(y) + ' '
                                    fd.write(line)
                            cv2.imwrite(save_path_img, image_show)
                    except Exception as e:
                        logger.error('Face landmark failed!')
                        logger.error(e)
                        sys.exit(-1)
                    else:
                        logger.info('Successful face landmark!')
