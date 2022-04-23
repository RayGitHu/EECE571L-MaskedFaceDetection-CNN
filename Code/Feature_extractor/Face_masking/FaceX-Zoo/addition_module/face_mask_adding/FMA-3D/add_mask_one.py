"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""
import os # ADDED
from face_masker import FaceMasker

if __name__ == '__main__':
    is_aug = False
    
    #12-25 & 27-31 & 34  ADDED  
    directory_path = 'EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Test_dataset/Test0/test'
    folders= os.listdir(directory_path) # list of all the folder names
    
    for folder in folders: # looping through all the folders
        if folder not in  ".DS_Store":
            folder_path= directory_path +"/"+ folder
            files= os.listdir(folder_path) # names of the files inside the folder
            files_count= len(files)      # number of the files inside the folder 

            to_directory= '/EECE571L-MaskedFaceDetection-CNN/Face_masking/FaceX-Zoo/face_sdk/api_usage/temp/add_mask_one/'+folder
            os.mkdir(to_directory)
            read_path_txt= '/EECE571L-MaskedFaceDetection-CNN/Face_masking/FaceX-Zoo/face_sdk/api_usage/temp/face_alignment/'+folder
            
            for file_ in files:
                if file_ not in ".DS_Store":
                    # read image
                    image_path = folder_path  +"/"+ file_
                    face_lms_file = read_path_txt +"/"+  file_  +'_' + 'landmark_res0.txt'

                    template_name = '7.png'
                    masked_face_path = to_directory +"/"+  file_   +'_mask.jpg'

                    face_lms_str = open(face_lms_file).readline().strip().split(' ')
                    face_lms = [float(num) for num in face_lms_str]
                    face_masker = FaceMasker(is_aug)
                    face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)
