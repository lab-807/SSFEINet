import time
import torch.distributed as dist
import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.io import savemat
from thop import profile
from thop import clever_format

#import Module
from models.SSFEINet import SSFEINet

from metrics import  calc_psnr as PSNR
from metrics import  calc_sam as SAM
from metrics import  calc_ergas as ERGAS
from metrics import  calc_rmse as RMSE
from metrics import  calc_cc as CC
from metrics import  calc_ssim as SSIM

from Utils import *
from dataset import cave_dataset
import option
opt = option.args_parser()
print(opt)


# Load testing data
print('===> Loading datasets')
key = 'Test.txt'
file_path = opt.test_data_path + key
file_list = loadpath(file_path)
HR_HSI, HR_MSI = prepare_data(opt.test_data_path,file_list, 12)
test_set = cave_dataset(opt, HR_HSI, HR_MSI, file_list, istrain=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False, pin_memory=True)

use_dist = opt.use_distribute
if use_dist:
    dist.init_process_group(backend="nccl", init_method='env://')

if use_dist:
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    local_rank = 0
    device = 'cuda:0'


if opt.arch == 'SSFEINet':
    model = SSFEINet(opt.scale_ratio,
                 opt.n_bands).cuda()

if use_dist:
    model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True,device_ids=[local_rank],output_device=local_rank)


psnr_total  = 0
sam_total   = 0
ergas_total = 0
ssim_total  = 0
rmse_total  = 0
cc_total    = 0

# Load the trained model parameters
model_out_path = opt.save_folder+"epoch_{}.pth".format(opt.arch)
if os.path.exists(model_out_path):
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    load_dict = torch.load(model_out_path, map_location=map_location)
    model.load_state_dict(load_dict['param'])

model.eval()

with torch.no_grad():
    for iteration, batch in enumerate(testing_data_loader, 1):
        Z, Y, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        
        Z = Variable(Z).float()
        Y = Variable(Y).float()
        X = Variable(X).float()
        img_name = batch[3][0]

        HX = model(Z,Y)
        HX = HX.clamp(min=0., max=1.)
        end_time = time.time()
        
        print(img_name)             
        X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
        HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()
        Z = torch.squeeze(Z).permute(1, 2, 0).cpu().numpy()
        
        rmse        = RMSE(X,HX)
        cc          = CC(X,HX)
        sam         = SAM(X,HX) 
        psnr        = PSNR(X,HX)
        ergas       = ERGAS(X,HX,opt.scale_ratio)
        ssim        = SSIM(X,HX)            

        psnr_total  += psnr
        sam_total   += sam
        ergas_total += ergas
        ssim_total  += ssim
        cc_total    += cc
        rmse_total  += rmse


print('Arch:   {}'.format(opt.arch))
print ('ModelSize(M):   {}'.format(np.around(os.path.getsize(model_out_path)//1024/1024.0, decimals=2)))
input1 = torch.randn((opt.test_batch_size, 31, 64, 64)).cuda()
input2 = torch.randn((opt.test_batch_size, 3, 512, 512)).cuda()
flops, params = profile(model, (input1, input2))
print(f"运算量：{flops/1e12}")
params = clever_format([params], '%.3f')
print(f"参数量：{params}")
print('params_bm:',sum([param.nelement() for param in model.parameters()])/ 1e6)

print("===> Avg. PSNR: {:.4f} dB".format(psnr_total / len(testing_data_loader)))
print("===> Avg. RMSE: {:.4f} ".format(rmse_total / len(testing_data_loader)))
print("===> Avg. ERGAS: {:.4f} ".format(ergas_total / len(testing_data_loader)))
print("===> Avg. SAM: {:.4f} ".format(sam_total/ len(testing_data_loader)))
print("===> Avg. SSIM: {:.4f} ".format(ssim_total / len(testing_data_loader)))
print("===> Avg. CC: {:.4f} ".format(cc_total / len(testing_data_loader)))      
    
