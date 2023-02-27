import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

channel_means = [0.598, 0.584, 0.565]
channel_stds  = [0.104, 0.103, 0.103]

def Bextraction(img):
    img = img[0].numpy()
    img1 = img.astype(np.uint8)
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)
    img2 =  cv2.dilate(img1, DIAMOND_KERNEL_5).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis = 0)
    return torch.tensor(img3.copy())


class ImgToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([transforms.ToTensor(),                                                                                    
                                 transforms.Normalize(channel_means, channel_stds)])
        return tf(img)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).long()


class Crack_loader(Dataset):
    """ dataset class for Crack datasets
    """
    
    def __init__(self, img_dir, img_fnames, mask_dir, mask_fnames, isTrain=False, resize=False):
        self.img_dir = img_dir
        self.img_fnames = img_fnames

        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames

        self.resize  = resize
        self.isTrain = isTrain

        self.aug = A.Compose([
                            A.augmentations.crops.transforms.RandomResizedCrop(256,256,p=0.5),
                            A.augmentations.MotionBlur(p=0.1),
                            A.augmentations.transforms.ColorJitter(),
                            A.augmentations.geometric.rotate.SafeRotate(),
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                            A.augmentations.geometric.rotate.RandomRotate90(p=0.5)
                            ])

        self.img_totensor  = ImgToTensor()

        self.mask_totensor = MaskToTensor()
                
    def __getitem__(self, i):
        # read a image given a random integer index
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = cv2.imread(fpath) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                    # H,W,3 np.uint8

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = cv2.imread(mpath, cv2.COLOR_BGR2GRAY)                                  # H,W, np.uint8

        if self.isTrain:
            img  = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)         # (256,256,3) np.uint8
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_CUBIC)        # (256,256) np.uint8

            # image augmentation     
            transformed = self.aug(image=img, mask=mask)
            img  = transformed['image']                                               # (256,256,3) np.uint8
            mask = transformed['mask']                                                # (256,256) np.uint8            
            
            # binarize segmentation
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
            # totensor
            img  = self.img_totensor(Image.fromarray(img.copy()))
            mask = self.mask_totensor(mask.copy()).unsqueeze(0)
        
            # extract boundary
            boundary = Bextraction(mask)                                              # (1,256,256) torch.float32

            return {'image': img,
                    'mask': mask,
                    'boundary': boundary}

        else:
            if self.resize:
                img  = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)     # (256,256,3) np.uint8
                mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_CUBIC)    # (256,256) np.uint8
                img  = self.img_totensor(Image.fromarray(img.copy()))
                
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            mask = self.mask_totensor(mask.copy()).unsqueeze(0)

            return {'image': img,
                    'mask': mask,
                    'img_path': fpath}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_fnames)