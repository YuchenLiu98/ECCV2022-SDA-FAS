# ECCV2022-SDA-FAS
The implementation of "Source-Free Domain Adaptation with Contrastive Domain Alignment and Self-supervised Exploration for Face Anti-Spoofing", ECCV2022.

The motivation of our proposed SDA-FAS:
<div align=center>
<img src="https://github.com/YuchenLiu98/ECCV2022-SDA-FAS/blob/main/imgs/motivation.PNG" width="450px">
</div>

The framework of our proposed SDA-FAS:
<div align=center>
<img src="https://github.com/YuchenLiu98/ECCV2022-SDA-FAS/blob/main/imgs/framework.PNG" width="750px">
</div>

## Congifuration Environment
- Python 3.7
- Pytorch 1.7.0 
- torchvision 0.8.1
- timm 0.3.2

## Data Preparation
### Dataset
Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, MSU-MFSD, and CelebA-Spoof datasets.
### Data Pre-processing.
MTCNN is used for face detection and alignment. All the cropped faces are resized as (256,256,3).
### Data Organization
```
└── Data_Dir
   ├── OULU_NPU
   ├── CASIA_MFSD
   ├── REPLAY_ATTACK
   ├── MSU_MFSD
   ├── CelebA-Spoof
   └── ...
```
## Training
Move to the folder $root/SDA-FAS/experiment/testing_scenarios/ and just run:
```
python train_SDAFAS.py
```
## Citation
Please cite our paper if the code is helpful to your research.
```
@inproceedings{liu2022source,
    author = {Liu, Yuchen and Chen, Yabo and Dai, Wenrui and Gou, Mengran and Huang, Chun-Ting and Xiong, Hongkai},
    title = {Source-Free Domain Adaptation with Contrastive Domain Alignment and Self-supervised Exploration for Face Anti-Spoofing},
    booktitle = {ECCV},
    year = {2022}
}

