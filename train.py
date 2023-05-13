import os
import random
from tqdm import trange

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import logging
from pathlib import Path

from model import ENet
from config import cfg
from loading_data import loading_data
from utils import calculate_mean_iu # TODO
from timer import Timer
import pdb

# LOGGING INITIALIZATION
cfg['LOGGING']['LOGGING_PATH'].mkdir(exist_ok=True,parents=True)
logging.basicConfig(
    filename=cfg['LOGGING']['LOGGING_PATH']/cfg['LOGGING']['LOGGING_FILENAME'], 
    filemode='w', 
    format='%(asctime)s\t%(levelname)s\t%(message)s',
    level=logging.DEBUG
)
logging.debug('initialized logging')

# LOGGI
exp_name = cfg['TRAIN']['EXP_NAME']
log_txt = cfg['TRAIN']['EXP_LOG_PATH'] + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg['TRAIN']['EXP_PATH']+ '/' + exp_name)
logging.debug('initialized tensorboard')

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()
logging.debug('initialized dataloader')


def main(loss_function=torch.nn.BCEWithLogitsLoss):

    with open('./config.py','r') as cfg_file, open(log_txt, 'a') as f:
        cfg_lines = cfg_file.readlines()
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if len(cfg['TRAIN']['GPU_ID'])==1:
        torch.cuda.set_device(cfg['TRAIN']['GPU_ID'][0])
    torch.backends.cudnn.benchmark = True
    
    if cfg['TRAIN']['STAGE']=='all':
        net = ENet(only_encode=False)
        if cfg['TRAIN']['PRETRAINED_ENCODER'] != '':
            encoder_weight = torch.load(cfg['TRAIN']['PRETRAINED_ENCODER'])
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg['TRAIN']['STAGE']=='encoder':
        net = ENet(only_encode=True)
    else: 
        net = ENet(only_encode=True)

    if len(cfg['TRAIN']['GPU_ID'])>1:
        net = torch.nn.DataParallel(net, device_ids=cfg['TRAIN']['GPU_ID']).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = loss_function().cuda() # Binary Classification
    optimizer = optim.Adam(net.parameters(), lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    scheduler = StepLR(optimizer, step_size=cfg['TRAIN']['NUM_EPOCH_LR_DECAY'], gamma=cfg['TRAIN']['LR_DECAY'])
    _t = {'train time' : Timer(),'val time' : Timer()} 
    # validate(val_loader, net, criterion, optimizer, -1, restore_transform)

    # TRAINING - VALIDATION LOOP
    for epoch_id, epoch in enumerate(trange(cfg['TRAIN']['MAX_EPOCH'])):

        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        # print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))

        _t['val time'].tic()
        iou_validation, loss_validation = validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)

        train_time = _t['train time'].diff
        validation_time = _t['val time'].diff
        logging_message = f'epoch: {epoch_id}\ttraining time: {train_time:.2f}s\tvalidation: time {validation_time:.2f}\tiou: {iou_validation:4f}\tiou: {loss_validation:4f}'
        print(logging_message)
        logging.info(logging_message)
        # print(f'epoch: {epoch_id}\tvalidation time: {train_time:.2f}s')


def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        # print('TRAIN','\t','outputs.shape',outputs.shape,'\t','labels',labels.unsqueeze(1).float().shape)
        loss.backward()
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    loss_ = 0.0

    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        #for binary classification
        outputs[outputs>0.5] = 1
        outputs[outputs<=0.5] = 0
        #for multi-classification ???

        iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
        # print('VALIDATION','\t','outputs.shape',outputs.shape,'\t','labels',labels.unsqueeze(1).float().shape)
        loss_ += criterion(outputs, labels.float())

    mean_iu = iou_/len(val_loader)
    mean_loss = loss_/len(val_loader)

    # print('[mean iu %.4f]' % (mean_iu)) 
    net.train()
    criterion.cuda()

    return mean_iu, mean_loss



if __name__ == '__main__':
    main()
