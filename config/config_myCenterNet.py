##########################仅需要修改如下部分##########################
train_cfg = dict(
    n_epoch=100,
    batch_size=16,
    lr=0.0001,
    dataloader_workers_num=12
)
test_cfg = dict(
     batch_size=16,
     post_process_workers_num=12
)
lnms_cfg = dict (
     kappa=0.6
)
dataset_cfg = dict(
    dataset_name='ROD2021',
    base_root="/media/wwb/NVME_CPU/ROD/data/cruw/test",
    data_root="/media/wwb/WIN10/ROD2021/sequences",
    anno_root="/media/wwb/WIN10/ROD2021/annotations",
    model_path_root="/log/checkpoint/",
    label_dir="/annotations/train/",
    snippet=4,
    process_num=24,
    resolving=(512,512),
#####################################################################
    anno_ext='.txt',
    train=dict(
        subdir='train',
    ),
    valid=dict(
        subdir='valid',
        seqs=[],
    ),
    test=dict(
        subdir='test',
    ),
    demo=dict(
        subdir='demo',
        seqs=[],
    ),
)



model_cfg = dict(
    type='CDC',
    name='rodnet-cdc-win16-wobg',
    max_dets=10,
    peak_thres=0.65,
    ols_thres=0.3,
)

confmap_cfg = dict(
    confmap_sigmas={
        'pedestrian': 15,
        'cyclist': 20,
        'car': 30,
        # 'van': 40,
        # 'truck': 50,
    },
    confmap_sigmas_interval={
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
        # 'van': [15, 40],
        # 'truck': [20, 50],
    },
    confmap_length={
        'pedestrian': 1,
        'cyclist': 2,
        'car': 3,
        # 'van': 4,
        # 'truck': 5,
    }
)



