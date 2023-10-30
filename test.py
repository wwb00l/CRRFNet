import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tools.dataloader import ourdataset
from tools.visualization import save_result_jpg
model=torch.load('./log/checkpoint/demo.pkl')
test_loader = DataLoader(ourdataset(False), batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
model.eval()
for i, (image,chirp,label) in enumerate(test_loader):
    with torch.no_grad():
        pred_temp=model(image.float().cuda(),chirp.float().cuda()).detach().cpu().numpy()
        save_result_jpg(i,pred_temp,image.detach().cpu().numpy(),
                     chirp.detach().cpu().numpy(),label.detach().cpu().numpy(),option="test")

