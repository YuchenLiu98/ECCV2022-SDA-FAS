import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
import math
import numpy as np
import imgaug.augmenters as iaa
import random
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])
def crop_face_from_scene_celebA(image,face_name_full, scale):
    real_h, real_w, _ = image.shape
    f=open(face_name_full,'r')
    lines=f.readlines()
    # print(face_name_full)
    haha = lines[0].split()
    y1 = int(int(haha[0])*(real_w / 224))
    x1 = int(int(haha[1])*(real_h / 224))
    w = int(int(haha[2])*(real_w / 224))
    h = int(int(haha[3])*(real_h / 224))
    f.close()
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)
    region=image[x1:x2,y1:y2]
    return region
def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region

class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img = sample
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img = sample
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return img

class FASDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    Cutout(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            bbox_path = img_path.split('.')[0]+'.dat'
            label = self.photo_label[item]
            img = Image.open(img_path)
            img1 = crop_face_from_scene(np.array(img), bbox_path, 1.0)
            img = Image.fromarray(img1)
            img = img.resize((256, 256))
            img_array = np.array(img)
            img_array = seq.augment_image(img_array)
            randomerasing = RandomErasing()
            img_array = randomerasing(img_array)
            img = Image.fromarray(img_array)
            img = self.transforms(img)
            return img, label
        else:
            img_path = self.photo_path[item]
            bbox_path = img_path.split('.')[0]+'.dat'
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img1 = crop_face_from_scene(np.array(img), bbox_path, 1.0)
            img = Image.fromarray(img1)
            img = img.resize((256, 256))
            img = self.transforms(img)
            return img, label, videoID

class FASDataset_Aug_both(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_label_target = data_pd['photo_label_target'].tolist()
        self.photo_confidence = data_pd['confidence'].tolist()
        self.photo_confidence_target = data_pd['confidence_target'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    Cutout(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            bbox_path = img_path.split('.')[0]+'.dat'
            label = self.photo_label[item]
            label_target = self.photo_label_target[item]
            img_confidence = self.photo_confidence[item]
            img_confidence_target = self.photo_confidence_target[item]
            img = Image.open(img_path)
            img1 = crop_face_from_scene(np.array(img), bbox_path, 1.0)
            img = Image.fromarray(img1)
            img = img.resize((256, 256))
            img_array = np.array(img)
            img_array1 = seq.augment_image(img_array)
            img_array2 = seq.augment_image(img_array)
            #randomerasing
            randomerasing = RandomErasing()
            img_array1 = randomerasing(img_array1)
            img_array2 = randomerasing(img_array2)
            img1 = Image.fromarray(img_array1)
            img2 = Image.fromarray(img_array2)
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            return img1, img2, label, label_target, img_confidence, img_confidence_target
        else:
            img_path = self.photo_path[item]
            bbox_path = img_path.split('.')[0]+'.dat'
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img1 = crop_face_from_scene(np.array(img), bbox_path, 1.0)
            img = Image.fromarray(img1)
            # img = img.resize((224, 224))
            img = img.resize((256, 256))
            img = self.transforms(img)
            return img, label, videoID
