from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import os
import random
import time
import socket
from torch.autograd import Variable
from torch.optim.lr_scheduler import  MultiStepLR
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

#Custom Module

from models.SSFEINet import SSFEINet


import option
from train import train
from loss import HybridLoss
from Utils import *
from dataset import cave_dataset

# ----------------------------------------
# Loading train set and set random seed
# ----------------------------------------
opt = option.args_parser()
print(opt)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(opt.seed)

# ----------------------------------------
# Building model and distribute model 
# Preparation  data
# ----------------------------------------
use_dist = opt.use_distribute
if use_dist:
    dist.init_process_group(backend="nccl", init_method='env://')

print('===> Loading datasets')
## Load training data
key = 'Train.txt'
file_path = opt.train_data_path + key
file_list = loadpath(file_path)
HR_HSI, HR_MSI = prepare_data(opt.train_data_path,file_list, 20)
train_set = cave_dataset(opt, HR_HSI, HR_MSI,file_list)

## Load testing data
key = 'Test.txt'
file_path = opt.test_data_path + key
file_list = loadpath(file_path)
HR_HSI, HR_MSI = prepare_data(opt.test_data_path,file_list, 12)
test_set = cave_dataset(opt, HR_HSI, HR_MSI, file_list, istrain=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False, pin_memory=True)

if use_dist:
    sampler = DistributedSampler(train_set)

if use_dist:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.train_batch_size, sampler = sampler, pin_memory=True)
else:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.train_batch_size, shuffle = True, pin_memory=True)


print('===> Building model')
print("===> distribute model")
if use_dist:
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    local_rank = 0
    device = 'cuda:0'

             
if opt.arch == 'SSFEINet':
    model = SSFEINet( opt.scale_ratio,
                  opt.n_bands).cuda()   

if use_dist:
    model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True,device_ids=[local_rank],output_device=local_rank)
print('Arch:   {}'.format(opt.arch))
print('Network parameters: {}'.format(sum(param.numel() for param in model.parameters()) / 1e6))
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=list(range(50,200,5)), gamma=0.95)

model_out_path = opt.save_folder+"epoch_{}.pth".format(opt.arch)

if os.path.exists(model_out_path):
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    load_dict = torch.load(model_out_path, map_location=map_location)
    opt.lr = load_dict['lr']
    opt.nEpochs = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])
    scheduler.load_state_dict(load_dict['scheduler'])
else:
    print("No saved model at {}, training will start from scratch ".format(opt.save_folder))

criterion = HybridLoss(weight=opt.alpha)
current_step = 0
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0        
mkdir(opt.save_folder)
mkdir(opt.outputpath)

def checkpoint(arch,epoch):

    model_out_path = opt.save_folder+"epoch_{}.pth".format(arch)
    if epoch % 1 == 0 and local_rank == 0:
        save_dict = dict(
            lr = optimizer.state_dict()['param_groups'][0]['lr'],
            param = model.state_dict(),
            adam = optimizer.state_dict(),
            epoch = epoch,
            scheduler = scheduler.state_dict()
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))


start_time = time.time()
print ('===>  Start Training: ')  
for epoch in range(opt.nEpochs + 1, 201):
    avg_loss = train(epoch, optimizer, model,training_data_loader,criterion,local_rank,tb_logger,current_step)
    checkpoint(opt.arch,epoch)
    torch.cuda.empty_cache()
    scheduler.step()
elapsed_time = time.time() - start_time
days = elapsed_time // (24 * 3600)
elapsed_time = elapsed_time % (24 * 3600)
hours = elapsed_time // (60 * 60)
minutes = (elapsed_time % 3600) // 60
print('Training complete in {:.0f}day {:.0f}h {:.0f}m '.format(days,hours,minutes))
