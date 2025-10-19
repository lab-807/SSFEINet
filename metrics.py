from skimage.metrics import structural_similarity
import numpy as np
import math


def calc_rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    rmse = math.sqrt(mse)
    return rmse

def calc_cc(ref, tar):
    # Get dimensions
    rows, cols,bands = tar.shape
    # Initialize output array
    out = np.zeros(bands)
    
    # Compute cross correlation for each band
    for i in range(bands):
        tar_tmp = tar[ :, :,i]
        ref_tmp = ref[ :, :,i]
        cc = np.corrcoef(tar_tmp.flatten(), ref_tmp.flatten())
        out[i] = cc[0, 1]

    return np.mean(out)


#PSNR for hyperspectral image ,multiple channels version 
def calc_psnr(img1, img2):
    ch = np.size(img1,2)
    if ch == 1:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        return s
    else:
        sum = 0
        for i in range(ch):
            mse = np.mean((img1[:,:,i] - img2[:,:,i]) ** 2)
            if mse == 0:
                return 100
            #The maximum pixel value for color images is 1
            PIXEL_MAX = np.max(img1[:,:,i])
            s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            sum = sum + s
        s = sum / ch
        return s


#ERGAS for hyperspectral image,multiple channels version,for eaxmple 512/64=8,ratio=8
def calc_ergas(img1,img2,ratio):

    I1 = img1.astype('float64')
    I2 = img2.astype('float64')

    Err = I1-I2

    ERGAS_index=0

    for iLR in range(I1.shape[2]):
        ERGAS_index = ERGAS_index + np.mean(Err[:,:,iLR]**2, axis=(0, 1))/(np.mean(I1[:,:,iLR], axis=(0, 1)))**2    

    ERGAS_index = (100/ratio) * math.sqrt((1/I1.shape[2]) * ERGAS_index)       
        
    return np.squeeze(ERGAS_index)


#calc_ssim for hyperspectral image
def calc_ssim(img1, img2):
    '''
    :param reference:
    :param target:
    :return:
    '''

    img1 = np.squeeze(img1)
    img1 = img1.reshape(img1.shape[0], -1)
    img2 = np.squeeze(img2)
    img2 = img2.reshape(img2.shape[0], -1)
    ssim = structural_similarity(img1, img2,data_range=1)

    return ssim

def  calc_sam(img1,img2):
    
    H = img2.shape[0]
    W = img2.shape[1]
    
    prod_scal = np.zeros((H,W))
    norm_orig = np.zeros((H,W))
    norm_fusa = np.zeros((H,W))
    for i in range(H):
        for j in range(W):            
            h1 = img1[i,j,:]
            h2 = img2[i,j,:]
            prod_scal[i,j] = h1.flatten() @ h2.flatten()
            norm_orig[i,j] = h1.flatten() @ h1.flatten()
            norm_fusa[i,j] = h2.flatten() @ h2.flatten()
    
    
    prod_norm = np.sqrt(norm_orig * norm_fusa)
    prod_map = prod_norm
    prod_map[prod_map == 0] = 2 * 10**(-16)
    
    prod_scal = np.reshape(prod_scal, (H*W,1))
    prod_norm = np.reshape(prod_norm, (H*W,1))
    
    z = np.nonzero(prod_norm == 0)
    
    prod_scal[z]=[]
    prod_norm[z]=[]
    
    angolo = np.sum(np.arccos(prod_scal/prod_norm))/(prod_norm.shape[0])
    
    SAM = angolo*180/math.pi
    
    return SAM

