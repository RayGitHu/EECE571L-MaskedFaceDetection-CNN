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


source /EECE571L-MaskedFaceDetection-CNN/Code/env_face/bin/activate

python -u /EECE571L-MaskedFaceDetection-CNN/Code/Similarity_analysis/Accuracy/Accuracy_mae_dot_ndot_Full_dataset_Without_FC3.py  --dataset_path /EECE571L-MaskedFaceDetection-CNN/Data/Similarity_analysis/Full_dataset --LocationofSavedModel /EECE571L-MaskedFaceDetection-CNN/Model/modelWithBestValidationAcc_unmasked_masked_250epochs_v4.h5
 




