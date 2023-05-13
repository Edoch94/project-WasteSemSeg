import os
import random

import torch
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model import ENet
# from configs.config_Enet import cfg
import configs.config_Enet
from loading_data import loading_data
import utils
from timer import Timer
import pdb

exp_name = configs.config_Enet.cfg['TRAIN']['EXP_NAME']
log_txt = configs.config_Enet.cfg['TRAIN']['EXP_LOG_PATH'] + '/' + exp_name + '.txt'
writer = SummaryWriter(configs.config_Enet.cfg['TRAIN']['EXP_PATH']+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()



save_best_model = utils.SaveBestModel()



def main():

    cfg_file = open(configs.config_Enet.__file__,"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(configs.config_Enet.cfg['TRAIN']['GPU_ID'])==1:
        torch.cuda.set_device(configs.config_Enet.cfg['TRAIN']['GPU_ID'][0])
    cudnn.benchmark = True

    net = []   
    
    if configs.config_Enet.cfg['TRAIN']['STAGE']=='all':
        net = ENet(only_encode=False)
        if configs.config_Enet.cfg['TRAIN']['PRETRAINED_ENCODER'] != '':
            encoder_weight = torch.load(configs.config_Enet.cfg['TRAIN']['PRETRAINED_ENCODER'])
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif configs.config_Enet.cfg['TRAIN']['STAGE'] =='encoder':
        net = ENet(only_encode=True)

    if len(configs.config_Enet.cfg['TRAIN']['GPU_ID'])>1:
        net = torch.nn.DataParallel(net, device_ids=configs.config_Enet.cfg['TRAIN']['GPU_ID']).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = torch.nn.BCEWithLogitsLoss().cuda() # Binary Classification
    optimizer = optim.Adam(net.parameters(), lr=configs.config_Enet.cfg['TRAIN']['LR'], weight_decay=configs.config_Enet.cfg['TRAIN']['WEIGHT_DECAY'])
    scheduler = StepLR(optimizer, step_size=configs.config_Enet.cfg['TRAIN']['NUM_EPOCH_LR_DECAY'], gamma=configs.config_Enet.cfg['TRAIN']['LR_DECAY'])
    _t = {'train time' : Timer(),'val time' : Timer()} 
    evaluate(val_loader, net, criterion, optimizer, -1, restore_transform)

    for epoch in range(configs.config_Enet.cfg['TRAIN']['MAX_EPOCH']):

        _t['train time'].tic()
        fit(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        
        _t['val time'].tic()
        val_iou, val_loss = evaluate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))
        print('VALIDATION LOSS')
        save_best_model(val_loss, epoch, net, optimizer, criterion)
        utils.save_model(epoch, net, optimizer, criterion)



def fit(train_loader, net, criterion, optimizer, epoch):
    net.train()
    criterion.cuda()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    # input_batches = []
    # output_batches = []
    # label_batches = []
    iou_ = 0.0
    loss_ = 0.0

    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        
        loss_ = criterion(outputs, labels.unsqueeze(1).float())
        #for binary classification
        outputs[outputs>0.5] = 1
        outputs[outputs<=0.5] = 0
        #for multi-classification ???

        iou_ += utils.calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
    mean_iu = iou_/len(val_loader)   
    mean_loss = loss_/len(val_loader)

    return mean_iu, mean_loss


if __name__ == '__main__':
    main()








