import glob
file_path=glob.glob('./brats18_dataset/train/**/*_flair.nii.gz',recursive=True)
print(len(file_path))
import random
import cv2
import numpy as np
import nibabel as nib
random.seed(10)
split_n=random.sample(range(285),85)
test_path=[file_path[i] for i in split_n]
train_path=list(set(file_path)-set(test_path))
split_n=random.sample(range(200),100)
gan_path=[train_path[i] for i in split_n]
train_path=list(set(train_path)-set(gan_path))
print('test',len(test_path))
print('train',len(train_path))
print('gan',len(gan_path))


flair=[]
t1=[]
t1ce=[]
t2=[]
seg=[]
tumor=0
nontumor=0
total=0
########################################################################################
for i in train_path:
    flair_path=i
    t1_path=i.replace('_flair','_t1')
    t1ce_path=i.replace('_flair','_t1ce')
    t2_path=i.replace('_flair','_t2')
    label_path=i.replace('_flair','_seg')

    img_flair=nib.load(flair_path).get_data()
    img_t1=nib.load(t1_path).get_data()
    img_t1ce=nib.load(t1ce_path).get_data()
    img_t2=nib.load(t2_path).get_data()
    label=nib.load(label_path).get_data()
    
    label[label!=0]=1
    
    img_flair_=img_flair.copy()
    img_flair_[img_flair_!=0]=1
    
    img_t1_=img_t1.copy()
    img_t1_[img_t1_!=0]=1
    
    img_t1ce_=img_t1ce.copy()
    img_t1ce_[img_t1ce_!=0]=1
    
    img_t2_=img_t2.copy()
    img_t2_[img_t2_!=0]=1
    
    for j in range(154,-1,-1):
        a=label[:,:,j].sum()
        b=img_flair_[:,:,j].sum()
        c=img_t1_[:,:,j].sum()
        d=img_t1ce_[:,:,j].sum()
        e=img_t2_[:,:,j].sum()
        threshold = 2000
        if b>threshold and c>threshold and d>threshold and e>threshold:

            img_slice=img_flair[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            flair.append(mm0)
            
            img_slice=img_t1[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t1.append(mm0)
            
            img_slice=img_t1ce[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t1ce.append(mm0)
            
            img_slice=img_t2[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t2.append(mm0)
            
            img_slice=label[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0[mm0>=0.5]=1.
            mm0[mm0!=1]=0.
            seg.append(mm0)
            
            total+=1
            if a>50:
                tumor+=1
            elif a==0:
                nontumor+=1                
# l_tumor.extend(l_nontumor)
np.save('./brats18_dataset/npy_train/train_flair.npy',flair)
np.save('./brats18_dataset/npy_train/train_t1.npy',t1)
np.save('./brats18_dataset/npy_train/train_t1ce.npy',t1ce)
np.save('./brats18_dataset/npy_train/train_t2.npy',t2)
np.save('./brats18_dataset/npy_train/train_seg.npy',seg)


flair=[]
t1=[]
t1ce=[]
t2=[]
seg=[]
tumor=0
nontumor=0
total=0
########################################################################################
for i in gan_path:
    flair_path=i
    t1_path=i.replace('_flair','_t1')
    t1ce_path=i.replace('_flair','_t1ce')
    t2_path=i.replace('_flair','_t2')
    label_path=i.replace('_flair','_seg')

    img_flair=nib.load(flair_path).get_data()
    img_t1=nib.load(t1_path).get_data()
    img_t1ce=nib.load(t1ce_path).get_data()
    img_t2=nib.load(t2_path).get_data()
    label=nib.load(label_path).get_data()
    
    label[label!=0]=1
    
    img_flair_=img_flair.copy()
    img_flair_[img_flair_!=0]=1
    
    img_t1_=img_t1.copy()
    img_t1_[img_t1_!=0]=1
    
    img_t1ce_=img_t1ce.copy()
    img_t1ce_[img_t1ce_!=0]=1
    
    img_t2_=img_t2.copy()
    img_t2_[img_t2_!=0]=1
    
    for j in range(154,-1,-1):
        a=label[:,:,j].sum()
        b=img_flair_[:,:,j].sum()
        c=img_t1_[:,:,j].sum()
        d=img_t1ce_[:,:,j].sum()
        e=img_t2_[:,:,j].sum()
        threshold = 2000
        if b>threshold and c>threshold and d>threshold and e>threshold:

            img_slice=img_flair[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            flair.append(mm0)
            
            img_slice=img_t1[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t1.append(mm0)
            
            img_slice=img_t1ce[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t1ce.append(mm0)
            
            img_slice=img_t2[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t2.append(mm0)
            
            img_slice=label[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0[mm0>=0.5]=1.
            mm0[mm0!=1]=0.
            seg.append(mm0)
            
            total+=1
            if a>50:
                tumor+=1
            elif a==0:
                nontumor+=1                
# l_tumor.extend(l_nontumor)
np.save('./brats18_dataset/npy_gan/gan_flair.npy',flair)
np.save('./brats18_dataset/npy_gan/gan_t1.npy',t1)
np.save('./brats18_dataset/npy_gan/gan_t1ce.npy',t1ce)
np.save('./brats18_dataset/npy_gan/gan_t2.npy',t2)
np.save('./brats18_dataset/npy_gan/gan_seg.npy',seg)


flair=[]
t1=[]
t1ce=[]
t2=[]
seg=[]
tumor=0
nontumor=0
total=0
########################################################################################
for i in test_path:
    flair_path=i
    t1_path=i.replace('_flair','_t1')
    t1ce_path=i.replace('_flair','_t1ce')
    t2_path=i.replace('_flair','_t2')
    label_path=i.replace('_flair','_seg')

    img_flair=nib.load(flair_path).get_data()
    img_t1=nib.load(t1_path).get_data()
    img_t1ce=nib.load(t1ce_path).get_data()
    img_t2=nib.load(t2_path).get_data()
    label=nib.load(label_path).get_data()
    
    label[label!=0]=1
    
    img_flair_=img_flair.copy()
    img_flair_[img_flair_!=0]=1
    
    img_t1_=img_t1.copy()
    img_t1_[img_t1_!=0]=1
    
    img_t1ce_=img_t1ce.copy()
    img_t1ce_[img_t1ce_!=0]=1
    
    img_t2_=img_t2.copy()
    img_t2_[img_t2_!=0]=1
    
    for j in range(154,-1,-1):
        a=label[:,:,j].sum()
        b=img_flair_[:,:,j].sum()
        c=img_t1_[:,:,j].sum()
        d=img_t1ce_[:,:,j].sum()
        e=img_t2_[:,:,j].sum()
        threshold = 2000
        if b>threshold and c>threshold and d>threshold and e>threshold:

            img_slice=img_flair[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            flair.append(mm0)
            
            img_slice=img_t1[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t1.append(mm0)
            
            img_slice=img_t1ce[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t1ce.append(mm0)
            
            img_slice=img_t2[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0=mm0/mm0.max()
            t2.append(mm0)
            
            img_slice=label[:,:,j]
            mm0=cv2.resize(img_slice,(128,128),interpolation=cv2.INTER_AREA)
            mm0[mm0>=0.5]=1.
            mm0[mm0!=1]=0.
            seg.append(mm0)
            
            total+=1
            if a>50:
                tumor+=1
            elif a==0:
                nontumor+=1                
# l_tumor.extend(l_nontumor)
np.save('./brats18_dataset/npy_test/test_flair.npy',flair)
np.save('./brats18_dataset/npy_test/test_t1.npy',t1)
np.save('./brats18_dataset/npy_test/test_t1ce.npy',t1ce)
np.save('./brats18_dataset/npy_test/test_t2.npy',t2)
np.save('./brats18_dataset/npy_test/test_seg.npy',seg)
