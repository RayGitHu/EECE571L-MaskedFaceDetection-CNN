#!/bin/bash
#SBATCH --account=def-panos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --mail-user=m.zidan@alumni.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%j-%x.log

module load StdEnv/2020 cuda/11 cudnn/8.0.3 llvm/8 python/3.7
export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$CUDA_HOME/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.0/cudnn/8.0.3/lib64
export LLVM_CONFIG=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/llvm/8.0.1/bin/llvm-config
export ENV_NAME=env_face


source /EECE571L-MaskedFaceDetection-CNN/Code/env_face/bin/activate  &&

cd /EECE571L-MaskedFaceDetection-CNN/Code/Feature_extractor/Face_masking/FaceX-Zoo/face_sdk &&
python3 api_usage/face_detect.py

cd /EECE571L-MaskedFaceDetection-CNN/Code/Feature_extractor/Face_masking/FaceX-Zoo/face_sdk &&
python3 api_usage/face_alignment.py




