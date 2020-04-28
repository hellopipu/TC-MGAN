# TC-MGAN
Pytorch implementation for multi-modality GAN with tumor-consistency loss for brain MR image synthesis

<p align='center'>
<img src='https://github.com/hellopipu/TC-MGAN/blob/master/doc/1.png' width='900'/>
</p>

Magnetic Resonance (MR) images of different modalities can provide complementary information for clinical diagno- sis, but whole modalities are often costly to access. Most existing methods only focus on synthesizing missing images between two modalities, which limits their robustness and efficiency when multiple modalities are missing. To address this problem, we propose a multi-modality generative adver- sarial network (MGAN) to synthesize three high-quality MR modalities (FLAIR, T1 and T1ce) from one MR modality T2 simultaneously. The experimental results show that the quality of the synthesized images by our proposed methods is better than the one synthesized by the baseline model, pix2pix. Besides, for MR brain image synthesis, it is impor- tant to preserve the critical tumor information in the generated modalities, so we further introduce a multi-modality tumor consistency loss to MGAN, called TC-MGAN. We use the synthesized modalities by TC-MGAN to boost the tumor segmentation accuracy, and the results demonstrate its effec- tiveness.

## Dependencies

- Python 3.6
- PyTorch 1.1.0

## Dataset
Download 285 cases training dataset of [BRATS18](https://www.med.upenn.edu/sbia/brats2018.html).

Then move all the 285 cases to ```./brats18_dataset/train/```

run the folowing command to pre-process, split and save the data as ```.npy``` files.
    ```bash
    python generate_data.py
    ```
The saved files are as below.
```
├── brats18_dataset
│   ├── npy_gan
│   │   └── gan_flair.npy
│   │   └── gan_t1ce.npy
│   │   └── gan_f2.npy
│   │   └── gan_t1.npy
│   ├── npy_test
│   │   └── test_flair.npy
│   │   └── test_t1ce.npy
│   │   └── test_f2.npy
│   │   └── test_t1.npy
│   ├── npy_train
│   │   └── train_flair.npy
│   │   └── train_t1ce.npy
│   │   └── train_f2.npy
│   │   └── train_t1.npy
```
## Train network
- Firstly, train a conditional unet which will be used for GAN training later.
    ```bash
    python train_unet_4.py
    ```
- Then, train the GAN model
    ```bash
    python train_gan.py
    ```
## Test network
    ```bash
    python test_gan.py
    ```
    
the generated images will be saved as below.
   ```
├── brats18_dataset
│   ├── npy_pred
│   │   └── pred_flair.npy
│   │   └── pred_t1ce.npy
│   │   └── pred_t1.npy

```
## Demo
You can download pretrained model here [[BaiduNet]](https://pan.baidu.com/s/1eg-bUKawINmy2B63pV_zCg)(code:wgne), and run the command below to try the demo.
    ```bash
    python demo.py
    ```
<p align='center'>
<img src='https://github.com/hellopipu/TC-MGAN/blob/master/doc/2.png' width='900'/>
</p>

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

If you find our code/models useful, please consider citing our paper:
```
@inproceedings{Xin2020Multi,
  author = {Xin, Bingyu and Hu, Yifan and Zheng, Yefeng and Liao, Hongen},
  title = {Multi-Modality Generative Adversarial Networks With Tumor Consistency Loss for Brain MR Image Synthesis},
  booktitle = {The IEEE International Symposium on Biomedical Imaging (ISBI)},
  year = {2020}
}
```

## Acknowledgments
- Codes borrowed a lot from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [StarGAN](https://github.com/yunjey/stargan).
