import torch
from torch.utils import data
from model import Unet
from utils import label2onehot,seed_torch
from dataset_brain import Dataset_gan
from loss import dice_loss
from loss import dice_score
seed_torch(10)
print('*******************train_pretrained_unet*******************')
file_path='./brats18_dataset/npy_gan/gan_t1.npy'
model_save='./weight/unet_pretrained.pth'
train_data=Dataset_gan(file_path)
batch_size=64
train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
unet=Unet()
optimizer=torch.optim.Adam(unet.parameters(),lr=0.0001)
unet.cuda()
unet.train()
EPOCH=30
for epoch in range(EPOCH):
    batch_score=0
    num_batch=0
    for i,(flair,t1,t1ce,t2,label) in enumerate(train_loader):
        info_c_ =torch.randint(3,(t1.size(0),))
        info_c = label2onehot(info_c_,3).cuda()
        img=torch.zeros(t1.size(0),t1.size(1),t1.size(2),t1.size(3))
        for i,l in enumerate(info_c_):
            if l==0:
                img[i]=flair[i]
            elif l==1:
                img[i] = t1ce[i]
            elif l==2:
                img[i] = t1[i]
        seg=unet(img.float().cuda(),info_c)

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


