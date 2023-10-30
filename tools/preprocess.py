import os 
from PIL import Image
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import Pool
from cruw import CRUW
from cruw.mapping import ra2idx
from cruw.visualization.draw_rf import magnitude
from tools.load_configs import load_configs_from_file
from tools.object_class import get_class_id
config_dict = load_configs_from_file(os.getcwd()+'/config/config_myCenterNet.py')
root_dir=config_dict['dataset_cfg']['base_root']
label_dir=root_dir+config_dict['dataset_cfg']['label_dir']
(pixel_x,pixel_y)=config_dict['dataset_cfg']['resolving']
process_num = config_dict['dataset_cfg']['process_num']

class_num=3
labeldict = {'pedestrian': 0, 'cyclist': 1, 'car': 2}
pack_data_dir="/pack_data_"+str(pixel_x)+"x"+str(pixel_y)
dataset = CRUW(data_root=root_dir, sensor_config_name='sensor_config_rod2021')

complib="blosc"#"bzip2"

confmap_sigmas = config_dict['confmap_cfg']['confmap_sigmas']
confmap_sigmas_interval = config_dict['confmap_cfg']['confmap_sigmas_interval']
confmap_length = config_dict['confmap_cfg']['confmap_length']
radar_configs = dataset.sensor_cfg.radar_cfg
range_grid = dataset.range_grid
angle_grid = dataset.angle_grid
n_class = dataset.object_cfg.n_class
classes = dataset.object_cfg.classes
snippet_num = config_dict['dataset_cfg']['snippet']


def count_sample(train2test):
    count=0
    if train2test=="train":
        trainimg_path=[]
        if os.path.exists(root_dir+pack_data_dir+"/")==True:
            for root,dirs,files in os.walk(root_dir+pack_data_dir):
                for file in files:
                    if "hdf" in file:
                        count+=int(file[-9:-4])-int(file[-15:-10])+1
            return count
        else:
            for root,dirs,files in os.walk(root_dir+"/sequences/train"): 
                for file in files: 
                    if "jpg" in file:
                        trainimg_path.append(os.path.join(root,file))
        count=len(trainimg_path)
        return count
    else:
        trainimg_path=[]
        for root,dirs,files in os.walk(root_dir+pack_data_dir+"_test"):
            for file in files:
                if "hdf" in file:
                    count+=int(file[-9:-4])-int(file[-15:-10])+1
    return count


def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range>0:
        return (data - np.min(data)) / _range
    return data

def read_hdf_data(index,train2test):
    if train2test=="train":
        for root,dirs,files in os.walk(root_dir+pack_data_dir):
            for file in files:
                if "hdf" in file and index>=int(file[-15:-10]) and index<=int(file[-9:-4]):
                    path=os.path.join(root,file)
                    chirp=np.array(pd.read_hdf(path,'c'+str(index).zfill(5))).reshape((128,128,8))
                    label=np.array(pd.read_hdf(path,'l'+str(index).zfill(5))).reshape((128,128,3))
                    image=np.array(pd.read_hdf(path,'i'+str(index).zfill(5))).reshape((512,512,3)).astype(np.float32)
                    return normalization(image),normalization(chirp),normalization(label)
    else:
        for root,dirs,files in os.walk(root_dir+pack_data_dir+"_test"):
            for file in files:
                if "hdf" in file and index>=int(file[-15:-10]) and index<=int(file[-9:-4]):
                    path=os.path.join(root,file)
                    chirp=np.array(pd.read_hdf(path,'c'+str(index).zfill(5))).reshape((128,128,8))
                    label=np.array(pd.read_hdf(path,'l'+str(index).zfill(5))).reshape((128,128,3))
                    image=np.array(pd.read_hdf(path,'i'+str(index).zfill(5))).reshape((512,512,3)).astype(np.float32)
                    return normalization(image),normalization(chirp),normalization(label)

def pack_batch(trainimg_path_batch):
    index_start=trainimg_path_batch[0][-5:]
    index_last=trainimg_path_batch[len(trainimg_path_batch)-1][-5:]
    name=trainimg_path_batch[0][-47:-29]
    h5 = pd.HDFStore(root_dir+pack_data_dir+"/"+name+"_"+index_start+"-"+index_last+"."+"hdf",'w',complevel=1, complib=complib)
    for i in range(int(index_last)-int(index_start)+1):
        image,chirp,label=load_Xtrain(trainimg_path_batch[i][:-5])
        h5['c'+str(int(index_start)+i).zfill(5)]=pd.Series(chirp.flatten())
        h5['i'+str(int(index_start)+i).zfill(5)]=pd.Series(image.flatten())
        h5['l'+str(int(index_start)+i).zfill(5)]=pd.Series(label.flatten())
    h5.close()


def load_Xtrain(trainimg_path):
    temp_path=trainimg_path.split('/')
    temp_path[-2]="RADAR_RA_H"
    temp_path[-1]=temp_path[-1][4:10]
    trainchirp_path="/".join(temp_path)
    trainlabel_path=label_dir+temp_path[-3]+".txt"
    image=np.asarray(Image.open(trainimg_path).resize((pixel_x,pixel_y)))
    chirp0=(normalization(np.load(trainchirp_path+"_0000.npy"))*255)
    chirp64=(normalization(np.load(trainchirp_path+"_0064.npy"))*255)
    chirp128=(normalization(np.load(trainchirp_path+"_0128.npy"))*255)
    chirp192=(normalization(np.load(trainchirp_path+"_0192.npy"))*255)
    chirp=np.concatenate((chirp0,chirp64,chirp128,chirp192),axis=2)
    #print(chirp.shape)
    label=np.zeros((3, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    df=pd.read_csv(trainlabel_path,sep=" ")
    for i in range(len(df)):
        if str(df.iloc[i,0]).zfill(6) == temp_path[-1][0:6]:
            class_id = get_class_id(df.iloc[i][3], classes)
            rng_idx, agl_idx = ra2idx(df.iloc[i][1], df.iloc[i][2], dataset.range_grid, dataset.angle_grid)
            sigma = 2 * np.arctan(confmap_length[df.iloc[i][3]] / (2 * range_grid[rng_idx])) * confmap_sigmas[df.iloc[i][3]]
            sigma_interval = confmap_sigmas_interval[df.iloc[i][3]]
            if sigma > sigma_interval[1]:
                sigma = sigma_interval[1]
            if sigma < sigma_interval[0]:
                sigma = sigma_interval[0]
            for i in range(radar_configs['ramap_rsize']):
                for j in range(radar_configs['ramap_asize']):
                    distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
                    if distant < 36:  # threshold for confidence maps
                        value = np.exp(- distant / 2) / (2 * math.pi)
                        label[class_id, i, j] = value if value > label[class_id, i, j] else label[class_id, i, j]

    label=np.rot90(np.transpose(label,(2, 1, 0)))
    return image,chirp,label
def find_subimage_num(path):
    count=0
    for root,dirs,files in os.walk(path): 
        for file in files: 
            if "jpg" in file:
                count+=1
    return count
'''
def pack_snippet_test(image,chirp,label,i):
    h5 = pd.HDFStore(root_dir+"/snippet_pack/test"+str(i).zfill(5)+".hdf",'w')#,complevel=1, complib="blosc")
    h5['c']=pd.Series(chirp.flatten().numpy())
    h5['i']=pd.Series(image.flatten().numpy())
    h5['l']=pd.Series(label.flatten().numpy())
    h5.close()

def pack_snippet_train(image,chirp,label,i):
    h5 = pd.HDFStore(root_dir+"/snippet_pack/train"+str(i).zfill(5)+".hdf",'w')#,complevel=1, complib="blosc")
    h5['c']=pd.Series(chirp.flatten().numpy())
    h5['i']=pd.Series(image.flatten().numpy())
    h5['l']=pd.Series(label.flatten().numpy())
    h5.close()
'''

def data_preprocess():
    if os.path.exists(root_dir+pack_data_dir+"/")==False:
        print("Data is being packaged, which will take a few minutes...")
        if os.path.exists(root_dir+pack_data_dir+"/")==False:
            os.makedirs(root_dir+pack_data_dir+"/") 
        trainimg_path=[]

        for root,dirs,files in os.walk(root_dir+"/sequences/train"): 
            for file in files: 
                if "jpg" in file:
                    trainimg_path.append(os.path.join(root,file))
        temp=trainimg_path[0][-47:-29]
        next_index=[]
        trainimg_path_batch=[]
        for i in range(len(trainimg_path)):
            trainimg_path[i]=trainimg_path[i]+str(i).zfill(5)
            if trainimg_path[i][-47:-29]!=temp:
                temp=trainimg_path[i][-47:-29]
                next_index.append(i)
        for i in range(len(next_index)-1):
            trainimg_path_batch.append(trainimg_path[next_index[i]:next_index[i+1]])
        trainimg_path_batch.append(trainimg_path[next_index[-1]:len(trainimg_path)])
        with Pool(process_num) as p:
            p.map(pack_batch, trainimg_path_batch)
        print("Done")

    '''
    from tqdm import tqdm
    from tools.dataloader import cruwdataset_train,cruwdataset_test
    from torch.utils.data import DataLoader

    if os.path.exists(root_dir+"/snippet_pack"+"/train00000.hdf")==False:
        print("Generate train snippet")
        train_loader = DataLoader(cruwdataset_train(), batch_size=1, shuffle=True,num_workers=int(config_dict['dataset_cfg']['process_num']/2))
        with tqdm(total=len(train_loader)) as t:
            p=Pool(int(config_dict['dataset_cfg']['process_num']/2))
            def update_t(result):
                t.update()
            for i, (image,chirp,label) in enumerate(train_loader):
                p.apply_async(pack_snippet_train, args = (image,chirp,label,i), callback = update_t)
            p.close()
            p.join()

    if os.path.exists(root_dir+"/snippet_pack"+"/test00000.hdf")==False:
        print("Generate test snippet")
        test_loader = DataLoader(cruwdataset_test(), batch_size=1, shuffle=False,num_workers=int(config_dict['dataset_cfg']['process_num']/2))
        with tqdm(total=len(test_loader)) as t:
            p=Pool(int(config_dict['dataset_cfg']['process_num']/2))
            def update_t(result):
                t.update()
            for i, (image,chirp,label) in enumerate(test_loader):
                p.apply_async(pack_snippet_test, args = (image,chirp,label,i), callback = update_t)
            p.close()
            p.join()
    '''
