import os
import random

import torch
from torch.backends import cudnn
from torch import optim
# from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

from model import ENet
# from configs.config_Enet import cfg
import configs.config_Enet
from loading_data import loading_data
import utils
# from timer import Timer
import pdb

exp_name = configs.config_Enet.cfg['TRAIN']['EXP_NAME']
log_txt = configs.config_Enet.cfg['TRAIN']['EXP_LOG_PATH'] + '/' + exp_name + '.txt'

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()



save_best_model = utils.SaveBestModel()



def main():

    # writer = SummaryWriter(configs.config_Enet.cfg['TRAIN']['EXP_PATH']+ '/' + exp_name)
    writer = SummaryWriter(comment=configs.config_Enet.cfg['TRAIN']['EXP_PATH']+ '_' + exp_name)

    _t = {'train time' : utils.Timer(), 'val time' : utils.Timer()} 

    cfg_file = open(configs.config_Enet.__file__,"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(configs.config_Enet.cfg['TRAIN']['GPU_ID'])==1:
        torch.cuda.set_device(configs.config_Enet.cfg['TRAIN']['GPU_ID'][0])
    cudnn.benchmark = True

    net = ENet()
    
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

    evaluate(val_loader, net, criterion)

    for epoch in range(configs.config_Enet.cfg['TRAIN']['MAX_EPOCH']):
        epoch+=1
        _t['train time'].tic()
        train_loss = fit(train_loader, net, criterion, optimizer)
        _t['train time'].toc(average=False)
        # print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        
        _t['val time'].tic()
        val_iou, val_loss = evaluate(val_loader, net, criterion)
        _t['val time'].toc(average=False)
        # print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))

        train_time = _t['train time'].diff
        val_time = _t['val time'].diff
        print(f'epoch: {epoch:04d}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, train_time {train_time:.4f}, val_time {val_time:.4f}')

        save_best_model(val_loss, epoch, net, optimizer, criterion)
        utils.save_model(epoch, net, optimizer, criterion)

        writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)

    writer.close()



def fit(train_loader, net, criterion, optimizer):
    net.train()
    criterion.cuda()

    loss_ = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        loss_ += criterion(outputs, labels.unsqueeze(1).float()).item()
    
    mean_loss = loss_/len(val_loader)

    return mean_loss


def evaluate(val_loader, net, criterion):
    net.eval()
    criterion.cpu()
    # input_batches = []
    # output_batches = []
    # label_batches = []
    iou_ = 0.0
    loss_ = 0.0

    with torch.no_grad():
        for vi, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            
            loss_ += criterion(outputs, labels.unsqueeze(1).float()).item()
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
