#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-panos
#SBATCH --job-name=face1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
module load arch/avx512 StdEnv/2018.3
nvidia-smi
 
  
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=yourEmail@ece.ubc.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load nixpkgs/16.09  gcc/7.3.0 opencv/4.2


source /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Code/env_face/bin/activate

python -u /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Code/Feature_extractor/training_code/train_face.py --dataDir /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Data/Feature_extractor/Unmasked_dataset --LocationofSavedModel /project/6003167/EECE571_2022/MaskedFaceRecognition_CNN/Github/EECE571L-MaskedFaceDetection-CNN/Model/modelOrgvgg19.h5  --numclasses 62 --epochs 100  
