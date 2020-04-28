import torch
from torch.utils import data
from unet import Unet
from dataset_brain import Dataset_brain_4
from loss import dice_loss
from loss import dice_score
torch.backends.cudnn.benchmark=True
torch.manual_seed(10)
file_path='./brats18_dataset/npy_test/test_t2.npy'
model_path='./weight/unet_4.pth'
train_data=Dataset_brain_4(file_path)
batch_size=64
train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,num_workers=4)
unet=Unet(4)
unet.load_state_dict(torch.load(model_path))
unet.cuda()
unet.eval()

batch_score=0
num_batch=0
with torch.no_grad():
    for i,(img,label) in enumerate(train_loader):
    #     print(img.shape)
        seg=unet(img.float().cuda())
        seg=seg.cpu()
        seg[seg>=0.5]=1.
        seg[seg!=1]=0.
        batch_score+=dice_score(seg,label.float()).data.numpy()
    #     print(num_batch)
        num_batch+=img.size(0)
        del seg,img,label
batch_score/=num_batch
print(' test_score = %.4f'%(batch_score))



