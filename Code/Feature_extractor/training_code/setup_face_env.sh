module load StdEnv/2020 cuda/11 cudnn/8.0.3 llvm/8 python/3.7
export LD_LIBRARY_PATH={$LD_LIBRARY_PATH}:$CUDA_HOME/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.0/cudnn/8.0.3/lib64
export LLVM_CONFIG=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/llvm/8.0.1/bin/llvm-config
export ENV_NAME=env_face


cd /EECE571L-MaskedFaceDetection-CNN/Code &&
rm -Rf /EECE571L-MaskedFaceDetection-CNN/Code/$ENV_NAME &&
virtualenv --no-download /EECE571L-MaskedFaceDetection-CNN/Code/$ENV_NAME &&
source /EECE571L-MaskedFaceDetection-CNN/Code/$ENV_NAME/bin/activate &&
pip install --no-index --upgrade pip &&

pip3 install opencv-python-headless pyyaml scikit-image pillow hdf5storage ninja scikit-learn onnx mxnet Cython &&
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchsummary==1.5.1 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html &&
pip3 install easydict timm==0.3.2 tensorboard



