from torch.utils import data
import numpy as np
class Dataset_gan(data.Dataset):
    def __init__(self,file):
        self.file_t1=np.load(file)
        file_seg=file.replace('t1','seg')
        file_flair=file.replace('t1','flair')
        file_t1ce=file.replace('t1','t1ce')
        file_t2=file.replace('t1','t2')
        self.label=np.load(file_seg)
        self.file_flair=np.load(file_flair)
        self.file_t1ce=np.load(file_t1ce)
        self.file_t2=np.load(file_t2)
    def __getitem__(self, index):
        flair=self.file_flair[index][np.newaxis,:]
        t1=self.file_t1[index][np.newaxis,:]
        t1ce=self.file_t1ce[index][np.newaxis,:]
        t2=self.file_t2[index][np.newaxis,:]
        flair=(flair-0.5)/0.5
        t1 = (t1 - 0.5) / 0.5
        t1ce = (t1ce - 0.5) / 0.5
        t2 = (t2 - 0.5) / 0.5
        label=self.label[index][np.newaxis,:]
        return flair,t1,t1ce,t2,label

    def __len__(self):
        return int(len(self.file_t1))

class Dataset_brain_4(data.Dataset):
    def __init__(self,file):
        self.file=np.load(file)
        file_seg='./brats18_dataset/npy_test/test_seg.npy'
      #file.replace('test_t2','test_seg').replace('npy_pred','npy_test')
        file_t1='./brats18_dataset/npy_pred/pred_t1.npy'
        #file.replace('test_t2','pred_t1')
        file_t1ce='./brats18_dataset/npy_pred/pred_t1ce.npy'
        #file.replace('test_t2','pred_t1ce')
        file_flair='./brats18_dataset/npy_pred/pred_flair.npy'
        #file.replace('test_t2','pred_flair')
        self.label=np.load(file_seg)
        self.file_t1=np.load(file_t1)
        self.file_t1ce=np.load(file_t1ce)
        self.file_flair=np.load(file_flair)
    def __getitem__(self, index):
        t2=self.file[index][np.newaxis,:]
        t2=(t2-0.5)/0.5
        t1=self.file_t1[index][np.newaxis,:]
        t1ce=self.file_t1ce[index][np.newaxis,:]
        flair=self.file_flair[index][np.newaxis,:]
        img=np.concatenate((flair,t1,t1ce,t2),axis=0)
        label=self.label[index][np.newaxis,:]
        return img,label
    def __len__(self):
        return len(self.file)

