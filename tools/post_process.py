import numpy as np
from tools.preprocess import config_dict,dataset
from tools.ols import get_ols_btw_objects
from tqdm import tqdm
from multiprocessing import Pool
import numba as nb
@nb.jit()
def detect_peaks(image, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            area = image[h - 1:h + 2, w - 2:w + 3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col

def lnms(obj_dicts_in_class):
    """
    Location-based NMS
    :param obj_dicts_in_class:
    :param config_dict:
    :return:
    """
    model_configs = config_dict['model_cfg']
    detect_mat = - np.ones((model_configs['max_dets'], 4))
    cur_det_id = 0
    # sort peaks by confidence score
    inds = np.argsort([-d['score'] for d in obj_dicts_in_class], kind='mergesort')
    dts = [obj_dicts_in_class[i] for i in inds]
    while len(dts) != 0:
        if cur_det_id >= model_configs['max_dets']:
            break
        p_star = dts[0]
        detect_mat[cur_det_id, 0] = p_star['class_id']
        detect_mat[cur_det_id, 1] = p_star['range_id']
        detect_mat[cur_det_id, 2] = p_star['angle_id']
        detect_mat[cur_det_id, 3] = p_star['score']
        cur_det_id += 1
        del dts[0]
        for pid, pi in enumerate(dts):
            ols = get_ols_btw_objects(p_star, pi, dataset)
            if ols > model_configs['ols_thres']:
                del dts[pid]

    return detect_mat

def detect(confmaps):

    batch_size, class_size,height, width = confmaps.shape
    n_class = dataset.object_cfg.n_class
    rng_grid = dataset.range_grid
    agl_grid = dataset.angle_grid
    model_configs = config_dict['model_cfg']
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']
    batch_size, class_size, height, width = confmaps.shape
    res_final = - np.ones((batch_size, max_dets, 4))

    for b in range(batch_size):
        detect_mat = []
        for c in range(class_size):
            obj_dicts_in_class = []
            confmap = np.squeeze(confmaps[b, c, :, :])
            rowids, colids = detect_peaks(confmap, threshold=peak_thres)

            for ridx, aidx in zip(rowids, colids):
                rng = rng_grid[ridx]
                agl = agl_grid[aidx]
                conf = confmap[ridx, aidx]
                obj_dict = {'frameid': None, 'range': rng, 'angle': agl, 'range_id': ridx, 'angle_id': aidx,
                                'class_id': c, 'score': conf}
                obj_dicts_in_class.append(obj_dict)

            detect_mat_in_class = lnms(obj_dicts_in_class)
            detect_mat.append(detect_mat_in_class)

        detect_mat = np.array(detect_mat)
        detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
        detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
        res_final[b, :, :] = detect_mat[:max_dets]

    return res_final

def detect_mul(pred,labels):

    class_size, height, width = labels.shape
    n_class = dataset.object_cfg.n_class
    rng_grid = dataset.range_grid
    agl_grid = dataset.angle_grid
    model_configs = config_dict['model_cfg']
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']
    pred_final = - np.ones((max_dets, 4))
    labels_final = - np.ones((max_dets, 4))

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap = np.squeeze(pred[c, :, :])
        rowids, colids = detect_peaks(confmap, threshold=peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap[ridx, aidx]
            obj_dict = dict(
                    frame_id=None,
                    range=rng,
                    angle=agl,
                    range_id=ridx,
                    angle_id=aidx,
                    class_id=c,
                    score=conf,
                    )
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = np.squeeze(lnms(obj_dicts_in_class))
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    pred_final[:, :] = detect_mat[:max_dets]

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap = np.squeeze(labels[c, :, :])
        rowids, colids = detect_peaks(confmap, threshold=peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap[ridx, aidx]
            obj_dict = dict(
                    frame_id=None,
                    range=rng,
                    angle=agl,
                    range_id=ridx,
                    angle_id=aidx,
                    class_id=c,
                    score=conf,
                    )
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = np.squeeze(lnms(obj_dicts_in_class))
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    labels_final[:, :] = detect_mat[:max_dets]

    return np.concatenate((pred_final,labels_final),axis=1)

def test_AP(pred,labels,ols=0.5):
    pbar = tqdm(total=len(pred))
    pbar.set_description('Postprocessing')

    def update_detect_mul(result):
        pred_final.append(result)
        pbar.update()

    pred_final=[]
    p=Pool(config_dict['test_cfg']['post_process_workers_num'])
    for i in range(len(pred)):
        p.apply_async(detect_mul, args = (pred[i],labels[i]), callback = update_detect_mul)
    p.close()
    p.join()

    from tools.ols import get_ap
    from functools import partial
    get_ap=partial(get_ap,ols=0.5)
    with Pool(config_dict['test_cfg']['post_process_workers_num']) as p:
        res=p.map(get_ap, pred_final)

    TP,FP,FN=np.sum(np.array(res),axis=0)
    AP=TP/(TP+FP)*100
    AR=TP/(TP+FN)*100
    return round(AP,2),round(AR,2)
