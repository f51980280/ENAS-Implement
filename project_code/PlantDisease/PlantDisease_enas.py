# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as trans
import torch
from torch.utils.data import random_split
from os import listdir
from os.path import join, splitext, basename
import glob
from torch.utils.data import DataLoader
from PIL import Image
from os import listdir
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image
from imgaug import augmenters as iaa
from matplotlib.pyplot import figure, imshow, axis
import imgaug as ia
import numpy as np
import statistics
import random
import natsort
import copy
import collections
import torchvision.models as models
import torch.optim as optim
#import datasets
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.nas.pytorch import enas
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)
from utils import accuracy, reward_accuracy
from util.cutout import Cutout
import warnings

logger = logging.getLogger('nni')

#To determine if your system supports CUDA
print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)

#Also can print your current GPU id, and the number of GPUs you can use.
print("Our selected device: ", torch.cuda.current_device())
print(torch.cuda.device_count(), " GPUs is available")


class Config():
    def __init__(self):
        self.FolderNames2English_names = {
                                    "Pepper__bell___Bacterial_spot":0, 
                                    "Pepper__bell___healthy":1,
                                    "Potato___Early_blight":2, 
                                    "Potato___healthy":3,
                                    "Potato___Late_blight":4, 
                                    "Tomato_Bacterial_spot":5,                               
                                    "Tomato_Early_blight":6,        
                                    "Tomato_healthy":7,           
                                    "Tomato_Late_blight":8,        
                                    "Tomato_Leaf_Mold":9,          
                                    "Tomato_Septoria_leaf_spot":10,
                                    "Tomato_Spider_mites_Two_spotted_spider_mite":11,
                                    "Tomato__Target_Spot":12,
                                    "Tomato__Tomato_mosaic_virus":13,
                                    "Tomato__Tomato_YellowLeaf__Curl_Virus":14,
                                    }
        self.folder_names2code = {}
        self.early_stop = 5
        self.max_epoch = 1000
        self.train_batchsize = 256
        self.eva_val_batchsize = 64
        self.class_num = 15
        self.each_class_item_num = {}
        self.temperature = 1
        self.alpha = 0.5
        
        self.image_size = 256  # resolution 

        
        self.train_dataset_path = r'/root/wei/DL_sys/Final_project/train'
        self.test_dataset_path = r'/root/wei/DL_sys/Final_project/validation'
        self.eval_dataset_path = r'/root/wei/DL_sys/Final_project/test'
        self.model_ouput_dir = './model/'
        
        self.save_name_rec = ''  #layer_resolution_groups_groupwidth
        self.best_epoch = 0
        
        self.net = 'MyResNeXt'  # 0: resnet18/MyResNeXt
        self.pretrain = False

        self.lr = 0.00001
        self.criterion = nn.CrossEntropyLoss()
        
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".JPG",".png", ".jpg", ".jpeg"])
        
class PlantDisease_Dataset(data.Dataset):
    def __init__(self, image_dir, 
                 input_transform, is_train=False, config=Config()):
        super(PlantDisease_Dataset, self).__init__()
        
        class_list = glob.glob(image_dir + '/*', recursive=True)
        
        self.datapath = image_dir
        self.image_filenames = []
        self.num_per_classes = {}
        self.class_dict_list = {}
        #print(len(files_list))
        
        for class_name in class_list:
            file_list = glob.glob(class_name + '/*', recursive=True)
            #print(class_name)
            
            for file in file_list:
                #print(file)
                self.image_filenames.append(file)
                class_name = class_name.split("/")[-1]
                if is_image_file(file):
                    if class_name in self.num_per_classes:
                        self.num_per_classes[class_name] += 1
                        self.class_dict_list[class_name].append(file)
                    else:
                        self.num_per_classes[class_name] = 1
                        self.class_dict_list[class_name] = []
                        self.class_dict_list[class_name].append(file)
        #print(self.num_per_classes)

        self.input_transform = input_transform
        
    def __getitem__(self, index):
        input_file = self.image_filenames[index]
        img = Image.open(input_file)
        img = img.convert('RGB')
        if self.input_transform is not None:
            img = self.input_transform(img)
            
        class_name = input_file.split('/')[-2]
        label = config.FolderNames2English_names[str(class_name)]
        return img, label

    def __len__(self):
        return len(self.image_filenames)
    
    def show_details(self):
        for key in sorted(self.num_per_classes.keys()):
            print("{:<50}|{:<20}|{:<5}".format(
                key,
                config.FolderNames2English_names[key],
                self.num_per_classes[key]
            ))
            
    def compute_weights(self):
        od_train_dataset_num_per_classes = self.num_per_classes.items()
        #print(list(od_train_dataset_num_per_classes)[0][1])
   
        wts = []
        prob = {}
        
        for i in range(len(list(od_train_dataset_num_per_classes))):
            if list(od_train_dataset_num_per_classes)[i][0] not in prob:
                prob[list(od_train_dataset_num_per_classes)[i][0]] = 1/list(od_train_dataset_num_per_classes)[i][1]
        
        for input_file in self.image_filenames :
            class_name = input_file.split('/')[-2]
            wts = np.append(wts,prob[class_name])
        
        print("Weighted Sample Ratio",wts,len(wts))
        return wts
        

class ImgAugTransform():
    def __init__(self, config=Config()):
        self.aug = iaa.Sequential([
            iaa.Scale((config.image_size, config.image_size)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),  iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)  
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)    

if __name__ == "__main__":
    
    parser = ArgumentParser("enas")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--search-for", choices=["macro", "micro"], default="macro")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

    config = Config()
    transform_train = trans.Compose([
        ImgAugTransform(config),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_train.transforms.append(Cutout(1,16))

    #The transform function for validation data
    transform_validation = trans.Compose([
        trans.Resize((config.image_size, config.image_size)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    #The transform function for test data
    transform_test = trans.Compose([
        trans.Resize((config.image_size, config.image_size)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    
    train_dataset = PlantDisease_Dataset(image_dir = config.train_dataset_path ,input_transform=transform_train, config=config)
    validation_dataset = PlantDisease_Dataset(image_dir = config.test_dataset_path ,input_transform=transform_validation, config=config)
    eval_dataset = PlantDisease_Dataset(image_dir = config.eval_dataset_path ,input_transform=transform_test, config=config)
    
    
    '''
    data_set = torchvision.datasets.ImageFolder(root = config.train_dataset_path ,transform=transform_train)
               
    val_eval_ratio = 0.1
    val_size = int(len(data_set) * val_eval_ratio)
    eval_size = int(len(data_set) * val_eval_ratio)
    train_val_size = len(data_set) - eval_size
    train_size = train_val_size - val_size
    
    tran_val_dataset , eval_dataset_split= random_split(data_set, [train_val_size, eval_size])
    train_dataset, validation_dataset = random_split(tran_val_dataset, [train_size, val_size])
    '''

   
    if args.search_for == "macro":
        model = GeneralNetwork()
        num_epochs = args.epochs or 310
        mutator = None
    elif args.search_for == "micro":
        model = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1, use_aux_heads=True)
        num_epochs = args.epochs or 200
        mutator = enas.EnasMutator(model, tanh_constant=1.1, cell_exit_extra_step=True)
    else:
        raise AssertionError

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
    print(len(train_dataset))
    print(len(validation_dataset))
    trainer = enas.EnasTrainer(model,
                               loss=criterion,
                               metrics=accuracy,
                               reward_function=reward_accuracy,
                               optimizer=optimizer,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                               batch_size=args.batch_size,
                               num_epochs=num_epochs,
                               dataset_train=train_dataset,
                               dataset_valid=validation_dataset,
                               log_frequency=args.log_frequency,
                               mutator=mutator)
    
    if args.visualization:
        trainer.enable_visualization()
    
    trainer.train()
