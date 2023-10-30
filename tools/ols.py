import math,os
import numpy as np
from cruw.mapping.coor_transform import pol2cart_ramap
from tools.object_class import get_class_name
from tools.load_configs import load_configs_from_file
from tools.preprocess import config_dict,dataset
from cruw.mapping.coor_transform import cart2pol_ramap

set_kappa=config_dict['lnms_cfg']['kappa']
threshold=0.4
def argmin(lst):
	return min(range(len(lst)), key=lst.__getitem__)

def find_nearest(point_list,point):
    dis_list=[]
    for i in range(len(point_list)):
        dis_list.append((point_list[i][0]-point[0])**2+(point_list[i][1]-point[1])**2)
    return argmin(dis_list)
def get_ap(data,ols=0):
    TP,FP,FN=0,0,0
    if ols==0:
        for i in range(50,95,5):
            TP_t,FP_t,FN_t=get_batch_correct_num(data,i/100)
            TP+=TP_t
            FP+=FP_t
            FN+=FN_t
        return TP/9,FP/9,FN/9
    else:
        return get_batch_correct_num(data,ols)

def get_batch_correct_num(data,ols_thres):
    pred, label=data[:,0:4],data[:,4:8]
    TP=0
    FN=0
    label_list=[]
    pred_list=[]
    for i in range(label.shape[0]):
        if label[i][3]>=threshold:
            label_list.append(label[i])
        else:
            break
    for i in range(pred.shape[0]):
        if pred[i][3]>=threshold:
            pred_list.append(pred[i])
        else:
            break
    pred_list=np.array(pred_list)
    label_list=np.array(label_list)
    for i in range(label_list.shape[0]):
        if pred_list.shape[0]==0:
            break
        nearest=find_nearest(pred_list[:,1:3],label_list[i,1:3])
        if int(pred_list[nearest][0])==int(label_list[i][0]):
            ols=get_ols_btw_pts(pred_list[nearest][1:3], label_list[i][1:3],int(label_list[i][0]), dataset)
            if ols>=ols_thres:
                TP+=1
            else:
                FN+=1
            pred_list[nearest][0]=-1
        else:
            FN+=1
    '''
    for i in range(pred_list.shape[0]):
        if label_list.shape[0]==0:
            FP+=pred_list.shape[0]
            break
        nearest=find_nearest(label_list[:,1:3],pred_list[i,1:3])
        if int(label_list[nearest][0])==int(pred_list[i][0]):
            ols=get_ols_btw_pts(label_list[nearest][1:3], pred_list[i][1:3],int(pred_list[i][0]), dataset)
            if ols<thres:
                FP+=1
        else:
            FP+=1
    '''
    FP=pred_list.shape[0]-TP
    return TP,FP,FN

def get_ols_btw_objects(obj1, obj2, dataset):
    classes = dataset.object_cfg.classes
    object_sizes = dataset.object_cfg.sizes
    if obj1['class_id'] != obj2['class_id']:
        print('Error: Computing OLS between different classes!')
        raise TypeError("OLS can only be compute between objects with same class.  ")
    if obj1['score'] < obj2['score']:
        raise TypeError("Confidence score of obj1 should not be smaller than obj2. "
                        "obj1['score'] = %s, obj2['score'] = %s" % (obj1['score'], obj2['score']))

    classid = obj1['class_id']
    class_str = get_class_name(classid, classes)
    rng1 = obj1['range']
    agl1 = obj1['angle']
    rng2 = obj2['range']
    agl2 = obj2['angle']
    x1, y1 = pol2cart_ramap(rng1, agl1)
    x2, y2 = pol2cart_ramap(rng2, agl2)
    dx = x1 - x2
    dy = y1 - y2
    s_square = x1 ** 2 + y1 ** 2
    kappa = object_sizes[class_str] / set_kappa
    e = (dx ** 2 + dy ** 2) / 2 / (s_square * kappa)
    ols = math.exp(-e)
    return ols

def get_ols_btw_pts(pt1, pt2, class_id, dataset):
    classes = dataset.object_cfg.classes
    object_sizes = dataset.object_cfg.sizes
    class_str = get_class_name(class_id, classes)
    rng_grid = dataset.range_grid
    agl_grid = dataset.angle_grid
    x1, y1 = pol2cart_ramap(rng_grid[int(pt1[0])], agl_grid[int(pt1[1])])
    x2, y2 = pol2cart_ramap(rng_grid[int(pt2[0])], agl_grid[int(pt2[1])])
    dx = x1 - x2
    dy = y1 - y2

    s_square = x1 ** 2 + y1 ** 2
    kappa = object_sizes[class_str] / set_kappa  # TODO: tune kappa
    e = (dx ** 2 + dy ** 2) / 2 / (s_square * kappa)
    ols = math.exp(-e)
    return ols
