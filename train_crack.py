from __future__ import division
import copy
import torch.optim as optim
from utils.utils import *
from pathlib import Path
from newloader import Crack_loader
from model.TransMUNet import TransMUNet
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


setup_seed(42)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_tra_path = config['path_to_tradata']
data_val_path = config['path_to_valdata']

DIR_IMG_tra  = os.path.join(data_tra_path, 'images')
DIR_MASK_tra = os.path.join(data_tra_path, 'masks')

DIR_IMG_val  = os.path.join(data_val_path, 'images')
DIR_MASK_val = os.path.join(data_val_path, 'masks')

img_names_tra  = [path.name for path in Path(DIR_IMG_tra).glob('*.jpg')]
mask_names_tra = [path.name for path in Path(DIR_MASK_tra).glob('*.png')]

img_names_val  = [path.name for path in Path(DIR_IMG_val).glob('*.jpg')]
mask_names_val = [path.name for path in Path(DIR_MASK_val).glob('*.png')]

train_dataset = Crack_loader(img_dir=DIR_IMG_tra, img_fnames=img_names_tra, mask_dir=DIR_MASK_tra, mask_fnames=mask_names_tra, isTrain=True)
valid_dataset = Crack_loader(img_dir=DIR_IMG_val, img_fnames=img_names_val, mask_dir=DIR_MASK_val, mask_fnames=mask_names_val, resize=True)
print(f'train_dataset:{len(train_dataset)}')
print(f'valiant_dataset:{len(valid_dataset)}')

train_loader  = DataLoader(train_dataset, batch_size = int(config['batch_size_tr']), shuffle= True,  drop_last=True)
val_loader    = DataLoader(valid_dataset, batch_size = int(config['batch_size_va']), shuffle= False, drop_last=True)

Net = TransMUNet(n_classes = number_classes)
flops, params = get_model_complexity_info(Net, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
print('flops: ', flops, 'params: ', params)
message = 'flops:%s, params:%s' % (flops, params)

Net = Net.to(device)
# load pretrained model
if int(config['pretrained']):
    Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
    best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']

optimizer = optim.Adam(Net.parameters(), lr= float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = config['patience'])
criteria  = DiceBCELoss()

# visual
visualizer = Visualizer(isTrain=True)
log_name = os.path.join('./checkpoints', config['loss_filename'])
with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)


for ep in range(int(config['epochs'])):
    # train
    Net.train()
    epoch_loss = 0
    for itter, batch in enumerate(train_loader):
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask'].to(device)
        boundary = batch['boundary'].to(device)
        mask_type = torch.float32 if Net.n_classes == 1 else torch.long
        msk = msk.to(device=device, dtype=mask_type)
        boundary = boundary.to(device=device, dtype=mask_type)
        msk_pred, B = Net(img,istrain=True)
        loss = criteria(msk_pred, msk)
        loss_boundary = criteria(B, boundary)
        tloss    = (0.8*(loss)) + (0.2*loss_boundary) 
        optimizer.zero_grad()
        tloss.backward()
        epoch_loss += tloss.item()
        optimizer.step()  
        if (itter+1)%int(float(config['progress_p']) * len(train_loader)) == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f' Epoch>> {ep+1} and itteration {itter+1} loss>>{epoch_loss/(itter+1)}')
        if (itter+1)*int(config['batch_size_tr']) == len(train_dataset):
            visualizer.print_current_losses(epoch=(ep+1), iters=(itter+1), loss=((epoch_loss/(itter+1))), lr=lr, isVal=False)


    # eval        
    with torch.no_grad():
        print('val_mode')
        val_loss = 0
        Net.eval()
        for itter, batch in enumerate(val_loader):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32 if Net.n_classes == 1 else torch.long
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred = Net(img)
            loss = criteria(msk_pred, msk)
            val_loss += loss.item()
        visualizer.print_current_losses(epoch=ep+1, loss=(abs(val_loss/(itter+1))), isVal=True)   
        mean_val_loss = (val_loss/(itter+1))
        # Check the performance and save the model
        if mean_val_loss < best_val_loss:
            best = ep + 1
            best_val_loss = copy.deepcopy(mean_val_loss)
            print('New best loss, saving...,best_val_loss=%6f' % (best_val_loss))
            with open(log_name, "a") as log_file:
                message = 'New best loss, saving...,best_val_loss=%6f' % (best_val_loss)
                log_file.write('%s\n' % message)
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})
            torch.save(state, config['saved_model'])

    scheduler.step(mean_val_loss)

    if ep+1 == int(config['epochs']): 
        visualizer.print_end(best, best_val_loss)
        state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})
        torch.save(state, config['saved_model_final'])

print('Trainng phase finished')    