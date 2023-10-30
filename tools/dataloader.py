from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os,cv2
from tools.preprocess import count_sample,root_dir,pack_data_dir
import numba as nb
def normalization(data):
    _range = torch.max(data) - torch.min(data)
    if _range>0:
        return (data - torch.min(data)) / _range
    return data
@nb.jit()
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn <= prob/2:
                output[i][j] = 0
            elif rdn > prob/2 and rdn <= prob:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
@nb.jit()
def sp_noise_crnjust(image,prob):
    output = np.zeros(image.shape,np.float32)
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            rdn = np.random.random()
            if rdn <= prob/2:
                output[:,i,j] = 0
            elif rdn > prob/2 and rdn <= prob:
                output[:,i,j] =1
            else:
                output[:,i,j] = image[:,i,j]
    return output

class ourdataset(Dataset):
    def __init__(self,train=True,noise_type="None",noise_intensity=0,noise_image=False,noise_radar=False,drop_image=False,drop_radar=False):
        self.hdf_list=[]
        self.total_num=0
        self.data_list=[]
        if train==True:
            self.path="./data/train/"
        else:
            self.path="./data/test/"
        self.train=train
        
        for root,dirs,files in os.walk(self.path):
            for file in files:
                if "hdf" in file:
                    self.hdf_list.append(self.path+file)
        for i in range(len(self.hdf_list)):
            self.total_num+=int(self.hdf_list[i][-9:-4])
            for j in range(int(self.hdf_list[i][-9:-4])):
                self.data_list.append((self.hdf_list[i],str(j).zfill(5)))

        self.drop_image_data=np.zeros((3,512,512))
        self.drop_radar_data=np.zeros((8,128,128))
        self.noise_type=noise_type
        self.noise_intensity=noise_intensity
        self.noise_image=noise_image
        self.noise_radar=noise_radar
        self.drop_image=drop_image
        self.drop_radar=drop_radar
    def __getitem__(self, index):
        if self.train==True:
            self.img=np.array(pd.read_hdf(self.data_list[index][0],"i"+self.data_list[index][1])).astype(np.float32).reshape((3,512,512))/255
            self.RF=np.array(pd.read_hdf(self.data_list[index][0],"f"+self.data_list[index][1])).astype(np.float32).reshape((8,128,128))
            self.heatmap=np.array(pd.read_hdf(self.data_list[index][0],"h"+self.data_list[index][1])).astype(np.float32).reshape((3,128,128))
            randnum=np.random.randint(1,4)
            if randnum==1:
                self.img=self.drop_image_data
            elif randnum==2:
                self.RF=self.drop_radar_data
            randnum=np.random.randint(1,4)
            if randnum==1:
                self.img=self.img+np.random.normal(loc=0,scale=0.1,size=self.img.shape)
            elif randnum==2:
                self.RF=self.RF+np.random.normal(loc=0,scale=0.1,size=self.RF.shape)
            return normalization(torch.from_numpy(self.img)),normalization(torch.from_numpy(self.RF)),normalization(torch.from_numpy(self.heatmap))

        else:
            self.img=np.array(pd.read_hdf(self.data_list[index][0],"i"+self.data_list[index][1])).astype(np.float32).reshape((3,512,512))/255
            self.RF=np.array(pd.read_hdf(self.data_list[index][0],"f"+self.data_list[index][1])).astype(np.float32).reshape((8,128,128))
            self.heatmap=np.array(pd.read_hdf(self.data_list[index][0],"h"+self.data_list[index][1])).astype(np.float32).reshape((3,128,128))
            return normalization(torch.from_numpy(self.img)),normalization(torch.from_numpy(self.RF)),normalization(torch.from_numpy(self.heatmap))

        
    def __len__(self):
        return self.total_num
