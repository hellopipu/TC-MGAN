import torch
from torch.utils import data
from dataset_brain import Dataset_gan
from model import netD,define_G,Unet
from utils import label2onehot,classification_loss,gradient_penalty,seed_torch,update_lr
from loss import dice_loss
import numpy as np
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
seed_torch()

print('*******************train_gan*******************')
file_path='./brats18_dataset/npy_gan/gan_t1.npy'

train_data=Dataset_gan(file_path)
batch_size=64

############fixed img
test_data=Dataset_gan('./brats18_dataset/npy_train/train_t1.npy')
fix_loader=data.DataLoader(dataset=test_data,batch_size=1,num_workers=4)
fix_iter=iter(fix_loader)
for i in range(40):
    next(fix_iter)
flair_fix,t1_fix,t1ce_fix,t2_fix,seg_fix=next(fix_iter)
origin_fix = np.hstack((t2_fix[0][0],flair_fix[0][0],t1ce_fix[0][0],t1_fix[0][0],seg_fix[0][0]))
for i in range(140):
    next(fix_iter)
flair_fix_2,t1_fix_2,t1ce_fix_2,t2_fix_2,seg_fix_2=next(fix_iter)
origin_fix_2 = np.hstack((t2_fix_2[0][0],flair_fix_2[0][0],t1ce_fix_2[0][0],t1_fix_2[0][0],seg_fix_2[0][0]))
for i in range(100):
    next(fix_iter)
flair_fix_3,t1_fix_3,t1ce_fix_3,t2_fix_3,seg_fix_3=next(fix_iter)
origin_fix_3 = np.hstack((t2_fix_3[0][0],flair_fix_3[0][0],t1ce_fix_3[0][0],t1_fix_3[0][0],seg_fix_3[0][0]))

del fix_loader,fix_iter,test_data
############

train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)

generator = define_G(4, 1, 64, 'unet_128', norm='instance', )
discriminator=netD()
unet = Unet()
unet.load_state_dict(torch.load("./weight/unet_pretrained.pth"))

optimizer_g=torch.optim.Adam(generator.parameters(),lr=0.0002)
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.0002)
optimizer_s=torch.optim.Adam(unet.parameters(),lr=0.0002)

generator.cuda()
discriminator.cuda()
unet.cuda()
EPOCH=100
num_iter=len(train_loader)
D_LOSS=[]
G_LOSS=[]
# S_LOSS=[]
f = open("./loss_gan.txt", 'a')
print(time.strftime('|---------%Y-%m-%d   %H:%M:%S---------|',time.localtime(time.time())),file=f)
discriminator.train()
unet.train()
for epoch in range(EPOCH):
    if epoch==30:
        update_lr(optimizer_g,0.0001)
        update_lr(optimizer_d,0.0001)
        update_lr(optimizer_s,0.0001)
        print('change lr to :',optimizer_g.param_groups[0]['lr'])
    elif epoch==60:
        update_lr(optimizer_g,0.00005)
        update_lr(optimizer_d,0.00005)
        update_lr(optimizer_s,0.00005)
        print('change lr to :',optimizer_g.param_groups[0]['lr'])
    elif epoch==90:
        update_lr(optimizer_g,0.00001)
        update_lr(optimizer_d,0.00001)
        update_lr(optimizer_s,0.00001)
        print('change lr to :',optimizer_g.param_groups[0]['lr'])
    d_loss_=0
    g_loss_=0
    d_loss_real_=0
    d_loss_cls_=0
    d_loss_fake_=0
    d_loss_gp_=0
    g_loss_fake_=0
    g_loss_cls_=0
    g_loss_rec_=0
    g_loss_seg_=0
    s_loss_ =0
    ##training mode set
    generator.train()
    for i,(flair,t1,t1ce,t2,seg) in enumerate(train_loader):
        ############################################## discriminator
        ############ real
        label_=torch.randint(3,(t1.size(0),))
        label = label2onehot(label_,3).cuda()
        real=torch.zeros(t1.size(0),t1.size(1),t1.size(2),t1.size(3))
        for i,l in enumerate(label_):
            if l==0:
                real[i]=flair[i]
            elif l==1:
                real[i] = t1ce[i]
            elif l==2:
                real[i] = t1[i]
            else:
                print('erro!!!')
        out_src, out_cls = discriminator(real.float().cuda(), t2.float().cuda())
        d_loss_real = - torch.mean(out_src.sum([1,2,3]))
        d_loss_cls = classification_loss(out_cls, label)

        ############################################## discriminator
        fake=generator(t2.float().cuda(),label)
        out_src, out_cls = discriminator(fake.detach(), t2.float().cuda())
        d_loss_fake = torch.mean(out_src.sum([1,2,3]))
        
        # Compute loss for gradient penalty.
        alpha = torch.rand(real.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * real.cuda().data + (1 - alpha) * fake.data).requires_grad_(True)
        out_src, _ = discriminator(x_hat,t2.float().cuda())
        d_loss_gp = gradient_penalty(out_src, x_hat)
#         d_loss_gp.backward(retain_graph=True)
        d_loss=d_loss_real+d_loss_fake+LAMBDA_CLS*d_loss_cls +LAMBDA_GP*d_loss_gp
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        ############################################## generator
        fake = generator(t2.float().cuda(), label)
        out_src,out_cls=discriminator(fake,t2.float().cuda())
        g_loss_fake = -torch.mean(out_src.sum([1,2,3]))
        g_loss_cls = classification_loss(out_cls,label)
        g_loss_rec = torch.mean(torch.abs(real.float().cuda() - fake).sum([1,2,3]))
        pred = unet(fake,label)
        g_loss_seg = dice_loss(pred,seg.float().cuda())
        g_loss = g_loss_fake + LAMBDA_CLS*g_loss_cls +  g_loss_rec*LAMBDA_REC + g_loss_seg*LAMBDA_SEG
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        ############################################## segmentor
        fake = generator(t2.float().cuda(), label)
        pred = unet(fake.detach(),label)
        pred_2 = unet(real.float().cuda(),label)
        s_loss = dice_loss(pred,seg.float().cuda())
        s_loss_2 = dice_loss(pred_2,seg.float().cuda())
        s_loss = 0.7*s_loss + 0.3 *s_loss_2
        optimizer_s.zero_grad()
        s_loss.backward()
        optimizer_s.step()
        
        
        d_loss_+=d_loss.data.cpu().numpy()
        g_loss_ += g_loss.data.cpu().numpy()
        d_loss_real_ +=d_loss_real.data.cpu().numpy()
        d_loss_cls_ +=d_loss_cls.data.cpu().numpy()
        d_loss_fake_ +=d_loss_fake.data.cpu().numpy()
        d_loss_gp_ +=d_loss_gp.data.cpu().numpy()
        g_loss_fake_ +=g_loss_fake.data.cpu().numpy()
        g_loss_cls_ +=g_loss_cls.data.cpu().numpy()
        g_loss_rec_+=g_loss_rec.data.cpu().numpy()
        g_loss_seg_+=g_loss_seg.data.cpu().numpy()
        s_loss_ += s_loss.data.cpu().numpy()

    print('EPOCH %d : d_loss = %.4f , g_loss = %.4f , s_loss = %.4f'%(epoch,d_loss_/num_iter,g_loss_/num_iter,s_loss_/num_iter))
    print(" d_real = %.4f , d_fake = %.4f , d_cls = %.4f , d_gp = %.4f |  g_fake = %.4f , g_cls = %.4f , g_rec = %.4f , g_seg = %.4f"%( d_loss_real_/num_iter , d_loss_fake_/num_iter , d_loss_cls_/num_iter , d_loss_gp.data.cpu().numpy() ,  g_loss_fake_/num_iter , g_loss_cls_/num_iter , g_loss_rec_/num_iter , g_loss_seg_/num_iter) )
    print("EPOCH %d : d_loss = %.4f , d_real = %.4f , d_fake = %.4f , d_cls = %.4f , d_gp = %.4f | g_loss = %.4f , g_fake = %.4f , g_cls = %.4f , g_rec = %.4f , g_seg = %.4f"%(epoch,d_loss_/num_iter , d_loss_real_/num_iter , d_loss_fake_/num_iter , d_loss_cls_/num_iter , d_loss_gp.data.cpu().numpy() , g_loss_/num_iter , g_loss_fake_/num_iter , g_loss_cls_/num_iter , g_loss_rec_/num_iter , g_loss_seg_/num_iter),file=f)
    D_LOSS.append(d_loss_/num_iter)
    G_LOSS.append(g_loss_ / num_iter)
#     S_LOSS.append(g_loss_seg_ / num_iter)
    #save plot fig
    x=[i for i in range(epoch+1)]
    plt.plot(x,G_LOSS,label='generator')
    plt.plot(x,D_LOSS,label='discriminator')
#     plt.plot(x,S_LOSS,label='segmentor')
    plt.legend()
    plt.grid(True)
    plt.savefig('gan_bw.png',format='png')
    plt.close()
    
    ##test for fixed
    generator.eval()
    c_fix=torch.tensor([[1,0,0]]).float().cuda()
    fix_flair = generator(t2_fix.float().cuda(),c_fix).data.cpu().numpy()
    fix_flair_2 = generator(t2_fix_2.float().cuda(),c_fix).data.cpu().numpy()
    fix_flair_3 = generator(t2_fix_3.float().cuda(),c_fix).data.cpu().numpy()
    
    c_fix=torch.tensor([[0,1,0]]).float().cuda()
    fix_t1ce = generator(t2_fix.float().cuda(),c_fix).data.cpu().numpy()
    fix_t1ce_2 = generator(t2_fix_2.float().cuda(),c_fix).data.cpu().numpy()
    fix_t1ce_3 = generator(t2_fix_3.float().cuda(),c_fix).data.cpu().numpy()
    
    c_fix=torch.tensor([[0,0,1]]).float().cuda()
    fix_t1 = generator(t2_fix.float().cuda(),c_fix).data.cpu().numpy()
    fix_t1_2 = generator(t2_fix_2.float().cuda(),c_fix).data.cpu().numpy()
    fix_t1_3 = generator(t2_fix_3.float().cuda(),c_fix).data.cpu().numpy()
    gen_fix = np.hstack((t2_fix[0][0],fix_flair[0][0],fix_t1ce[0][0],fix_t1[0][0],seg_fix[0][0]))
    gen_fix_2 = np.hstack((t2_fix_2[0][0],fix_flair_2[0][0],fix_t1ce_2[0][0],fix_t1_2[0][0],seg_fix_2[0][0]))
    gen_fix_3 = np.hstack((t2_fix_3[0][0],fix_flair_3[0][0],fix_t1ce_3[0][0],fix_t1_3[0][0],seg_fix_3[0][0]))
    
    plt.axis('off')
    plt.imshow(np.vstack((origin_fix,gen_fix,origin_fix_2,gen_fix_2,origin_fix_3,gen_fix_3)) )
    plt.savefig('glips_bw.png',format='png')
    plt.close()

    
    
f.close()




model_save_g='./weight/generator_t2_tumor_bw.pth'
model_save_g='./weight/segmentor_t2_tumor_bw.pth'
model_save_d='./weight/discriminator_t2_bw.pth'
torch.save(generator.state_dict(),model_save_g)
torch.save(unet.state_dict(),model_save_g)
torch.save(discriminator.state_dict(),model_save_d)








