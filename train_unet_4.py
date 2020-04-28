import torch
from torch.utils import data
from unet import Unet
from dataset_brain import Dataset_brain_4
from utils import seed_torch
from loss import dice_loss
from loss import dice_score

seed_torch()
print('*******************train_unet_4*******************')
file_path='./brats18_dataset/npy_train/train_flair.npy'
model_save='/weight/unet_4.pth'
train_data=Dataset_brain_4(file_path)
batch_size=64
train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
unet=Unet(4)
optimizer=torch.optim.Adam(unet.parameters(),lr=0.001)
unet.cuda()
unet.train()
EPOCH=30
print(EPOCH)
for epoch in range(EPOCH):
    batch_score=0
    num_batch=0
    for i,(img,label) in enumerate(train_loader):
        seg=unet(img.float().cuda())
        loss=dice_loss(seg,label.float().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        seg=seg.cpu()
        seg[seg>=0.5]=1.
        seg[seg!=1]=0.
        batch_score+=dice_score(seg,label.float()).data.numpy()
        num_batch+=img.size(0)
    batch_score/=num_batch
    print('EPOCH %d : train_score = %.4f'%(epoch,batch_score))
torch.save(unet.state_dict(),model_save)


