import numpy as np
import matplotlib.pyplot as plt
import torch
from model import define_G

## load sample
a=np.load('./brats18_dataset/sample.npy').item()
t2 = ((torch.from_numpy(a['t2'])[np.newaxis,np.newaxis,:]-0.5)/0.5).float().cuda()
## load model
model_path = './weight/generator_t2_tumor.pth'
gen = define_G(4, 1, 64, 'unet_128', norm='instance', )
gen.load_state_dict(torch.load(model_path))
gen.cuda()
gen.eval()

## predict flair using t2
c=torch.zeros(1,3).cuda()
c[np.arange(t2.size(0)),0]=1
f_pred = (gen(t2,c).squeeze().data.cpu().numpy()+1)/2

## predict t1ce using t2
c=torch.zeros(1,3).cuda()
c[np.arange(t2.size(0)),1]=1
t1ce_pred = (gen(t2,c).squeeze().data.cpu().numpy()+1)/2

## predict t1ce using t2
c=torch.zeros(1,3).cuda()
c[np.arange(t2.size(0)),2]=1
t1_pred = (gen(t2,c).squeeze().data.cpu().numpy()+1)/2

## plot img
img1 = np.hstack((a['t2'],a['flair'],a['t1'],a['t1ce'],a['seg']))
img2 = np.hstack((a['t2'],f_pred,t1_pred,t1ce_pred,a['seg'],))
img = np.vstack((img1,img2))
plt.rcParams['figure.figsize'] = (15, 15)
plt.imshow(img,'gray')
plt.axis('off')
plt.show()#
#plt.savefig('sample2.png',format='png')