## NNI & ENAS Reference
https://arxiv.org/abs/1802.03268  
https://github.com/microsoft/nni

## Environment Install  
##### NNI supports and is tested on Ubuntu >= 16.04, macOS >= 10.14.1, and Windows 10 >= 1809. Simply run the following pip install in an environment that has python 64-bit >= 3.5.  

### Recommend anaconda environment for the project
## Device
```
OS: Ubuntu 16.04
GPU: NVIDIA Tesla V100 32GB
cuda version:10.2
python : 3.7.2
```
## Linux or MacOS
```
 apt-get update  
 python3 -m pip install --upgrade nni  
 conda install -c pytorch torchvision cudatoolkit=10.2 pytorch
 pip3 install imgaug  
 pip3 install opencv-python  
 apt-get install -y libsm6 libxext6 libxrender-dev   
 pip3 install natsort  
```
## File & Folder Discription
##### PlantDisease : We train the enas in macro or micro search space in this folder
##### model : model's saving path
##### train : training data set
##### test : testing data set
##### validation : validation data set
##### train_base_net_one_gpu.ipynb :
##### trainer.py : Substitute the enas trainer in your python/site-packages/nni/nas/pytorch path










