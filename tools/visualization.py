import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from multiprocessing import Pool
from functools import partial
from tools.post_process import detect
from tools.load_configs import load_configs_from_file

def save_result_jpg(now_batch,pred,img,chirp,label,option="train"):
    labeldict = {0:'pedestrian', 1:'cyclist', 2:'car'}
    pred_final=detect(pred)
    label_final=detect(label)
    pred=np.transpose(pred*255,(0,3,2,1)).astype('uint8')
    img=np.transpose(img*255,(0,3,2,1)).astype('uint8')
    chirp=np.transpose(chirp*255,(0,3,2,1)).astype('uint8')
    label=np.transpose(label*255,(0,3,2,1)).astype('uint8')
    batchsize=len(pred)
    target = Image.new('RGB', (216*2+720,432))
    for i in range(batchsize):

        img_pred=Image.fromarray(pred[i]).convert('RGB').resize((216,216))
        draw_pred = ImageDraw.Draw(img_pred)       
        img_label=Image.fromarray(label[i]).convert('RGB').resize((216,216))
        draw_label = ImageDraw.Draw(img_label) 

        img_chirp1=Image.fromarray(chirp[i,:,:,0]).convert('RGB').resize((216,216))
        img_chirp2=Image.fromarray(chirp[i,:,:,1]).convert('RGB').resize((216,216))
        img_img=Image.fromarray(img[i]).convert('RGB').resize((720,432))
        target.paste(img_pred, (0,0))
        target.paste(img_label, (216,0))
        target.paste(img_chirp1, (0,216))
        target.paste(img_chirp2, (216,216))
        target.paste(img_img, (216*2,0))
        if option=="train":
            target.save(os.getcwd()+"/log/img/train/"+str(now_batch*batchsize+i).zfill(6)+".jpg")
        if option=="test":
            target.save(os.getcwd()+"/log/img/test/"+str(now_batch*batchsize+i).zfill(6)+".jpg")
    target.close()

