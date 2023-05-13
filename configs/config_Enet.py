import os
# from easydict import EasyDict as edict
import time
import torch


# init
__C = {}

cfg = __C
__C['DATA'] = {}
__C['NET'] = {}
__C['TRAIN'] = {}
__C['VAL'] = {}
__C['TEST'] = {}
__C['VIS'] = {}
__C['MODEL'] = {}

#------------------------------MODEL-----------------------
__C['MODEL']['NAME'] = 'Enet'

#------------------------------DATA------------------------

__C['DATA']['DATASET'] = 'city'  # dataset # city
__C['DATA']['DATA_PATH'] = 'dataset'
__C['DATA']['NUM_CLASSES'] = 1
__C['DATA']['IGNORE_LABEL'] = 255
__C['DATA']['IGNORE_LABEL_TO_TRAIN_ID'] = 19 # 255->19
                                          

__C['DATA']['MEAN_STD'] = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#------------------------------TRAIN------------------------

# stage
__C['TRAIN']['STAGE'] = 'encoder' # encoder or all
__C['TRAIN']['PRETRAINED_ENCODER'] = '' # Path of the pretrained encoder

# input setting
__C['TRAIN']['BATCH_SIZE'] = 16 #imgs
__C['TRAIN']['IMG_SIZE'] = (224,448)

__C['TRAIN']['GPU_ID'] = [0]


__C['TRAIN']['RESUME'] = ''#model path

# learning rate settings
__C['TRAIN']['LR'] = 5e-4
__C['TRAIN']['LR_DECAY'] = 0.995
__C['TRAIN']['NUM_EPOCH_LR_DECAY'] = 1 #epoches

__C['TRAIN']['WEIGHT_DECAY'] = 2e-4

__C['TRAIN']['MAX_EPOCH'] = 5

# output 
__C['TRAIN']['PRINT_FREQ'] = 10
__C['TRAIN']['FINAL_MODEL_SAVE_EPOCH_FREQ'] = 1

now = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())

__C['TRAIN']['EXP_NAME'] =  now \
                    + '_' + __C['TRAIN']['STAGE'] \
                    + '_' + __C['MODEL']['NAME']  \
                    + '_' + __C['DATA']['DATASET'] \
                    + '_' + str(__C['TRAIN']['IMG_SIZE']) \
                    + '_lr_' + str(__C['TRAIN']['LR'])


__C['TRAIN']['LABEL_WEIGHT'] = torch.FloatTensor([1,1])

__C['TRAIN']['CKPT_NETFOLDER_PATH'] = os.path.join('./ckpt', __C['MODEL']['NAME'])
__C['TRAIN']['EXP_LOG_PATH'] = os.path.join('./logs', __C['MODEL']['NAME'])
__C['TRAIN']['EXP_PATH'] = os.path.join('./exp', __C['MODEL']['NAME'])
__C['TRAIN']['CKPT_MODEL'] = os.path.join(__C['TRAIN']['CKPT_NETFOLDER_PATH'], __C['TRAIN']['EXP_NAME'])

#------------------------------VAL------------------------
__C['VAL']['BATCH_SIZE'] = 16 # imgs
__C['VAL']['SAMPLE_RATE'] = 1

#------------------------------TEST------------------------
__C['TEST']['GPU_ID'] = 0

#------------------------------VIS------------------------

__C['VIS']['SAMPLE_RATE'] = 0

__C['VIS']['PALETTE_LABEL_COLORS'] = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


#------------------------------MISC------------------------
# mkdir for the training CKPT 
if not os.path.exists(__C['TRAIN']['CKPT_NETFOLDER_PATH']):
    os.mkdir(__C['TRAIN']['CKPT_NETFOLDER_PATH'])
if not os.path.exists(__C['TRAIN']['CKPT_MODEL']):
    os.mkdir(__C['TRAIN']['CKPT_MODEL'])

# mkdir for the training log
if not os.path.exists(__C['TRAIN']['EXP_LOG_PATH']):
    os.mkdir(__C['TRAIN']['EXP_LOG_PATH'])

# mkdir for the training exp
if not os.path.exists(__C['TRAIN']['EXP_PATH']):
    os.mkdir(__C['TRAIN']['EXP_PATH'])

#================================================================================
#================================================================================
#================================================================================  
