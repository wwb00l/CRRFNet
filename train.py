import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tools.dataloader import ourdataset
from tools.visualization import save_result_jpg
from Models.myCenterNet import myCenterNet
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm
model=myCenterNet(class_num=3).float().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
criterion=nn.MSELoss().cuda()
train_loader = DataLoader(ourdataset(), batch_size=32,shuffle=True,num_workers=24,pin_memory=True)
model.train()
scaler = GradScaler()
for epoch in range(0,100):
    mloss=0
    loss=0
    print('Epoch:' ,epoch)
    with tqdm(total=len(train_loader)) as t:
        t.set_description('Training')
        for i, (image,chirp,label) in enumerate(train_loader):
            optimizer.zero_grad()
            label=label.float().cuda()
            chirp=chirp.float().cuda()
            image=image.float().cuda()
            with autocast():
                pred = model(image,chirp)
                loss = criterion(pred, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            mloss = (mloss * i + loss.item()) / (i + 1)
            t.set_postfix({'loss' : '{0:1.7f}'.format(mloss)})
            t.update(1)
        torch.save(model, './log/checkpoint/'+str(epoch).zfill(3)+'.pkl')
        t.close()
