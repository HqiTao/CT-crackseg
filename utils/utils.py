import cv2
import os
import time
import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

config = yaml.load(open('./config_crack.yml'), Loader=yaml.FullLoader)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_img_patches(img):

    img_height, img_width, _ = img.shape

    input_height = input_width = 256

    stride_ratio = 0.5
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []

    if img_height < img_width:
        assert img_height < 2*input_height
        y_corner = [0, img_height-input_height, int(0.5*(img_height-input_height))]
        for y in y_corner:
            for x in range(0, img_width - input_width + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
            if x != img_width - input_width:
                x = img_width - input_width
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
    else:
        assert img_width < 2*input_width
        x_corner = [0, img_width-input_width, int(0.5*(img_width-input_width))]
        for x in x_corner:
            for y in range(0, img_height - input_height + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
            if y != img_height - input_height:
                y = img_height - input_height
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
    
    assert np.all(normalization_map >= 1)

    patches.append(cv2.resize(img, (input_height, input_width), interpolation=cv2.INTER_CUBIC))

    patches = np.array(patches)
    
    return patches, patch_locs


def merge_pred_patches(img, preds, patch_locs):

    img_height, img_width, _ = img.shape

    input_height = input_width = 256

    probability_map = np.zeros((img_height, img_width), dtype=float)
    num1 = np.zeros((img_height, img_width), dtype=np.int16)
    
    for i, response in enumerate(preds):
        if i < len(preds)-1:
            coords = patch_locs[i]
            probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response
            num1[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += 1
        else:
            mskp = cv2.resize(response, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    assert np.all(num1 != 0)
    probability_map = probability_map / num1

    msk_pred = 0.5*probability_map + 0.5 * mskp

    return msk_pred


class Visualizer():
    def __init__(self, isTrain=False):
        self.log_name = os.path.join('./checkpoints', config['loss_filename'])
        self.isTrain = isTrain
        if self.isTrain:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training loss (%s) ================\n' % now)
        else:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing begin (%s) ================\n' % now)

    def print_current_losses(self, epoch=0, iters=0, loss=0.,  lr=0., isVal=False):
        """print current losses on console; also save the losses to the disk
        """
        if not isVal: # train
            message = '(epoch: %d, iters: %d) mean_loss: %6f lr: %6f' % (epoch, iters, loss, lr)
            print(message)  # print the message
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
        elif isVal: # val
            message = 'validation on epoch>> %d, mean tloss>> %6f ' % (epoch, loss)
            print(message)  # print the message
            with open(self.log_name, "a") as log_file:
                log_file.write('val_mode:%s\n' % message)  # save the message

    def print_end(self, best=0, best_val_loss=0.):
        message = 'best model appear in epoch%d and best val_loss is %6f' % (best, best_val_loss)
        end_now = time.strftime("%c")
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            log_file.write('================ Training End (%s) ================\n' % end_now)

    def print_val(self, tn, fp, fn, tp, precision, recll, f1):
        message = 'TN=%d, FP= %d, FN=%d, TP=%d\nprecision:%6f, recall:%6f, F1_score:%6f' % (tn, fp, fn, tp, precision, recll, f1)
        end_now = time.strftime("%c")
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            log_file.write('================ Testing End (%s) ================\n' % end_now)
                

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weight=torch.zeros_like(targets)
        weight=torch.fill_(weight, 0.04)
        weight[targets>0]=0.96
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean', weight=weight)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE