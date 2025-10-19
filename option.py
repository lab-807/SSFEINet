import argparse


# ----------------------------------------
# Training settings
# ----------------------------------------
def args_parser():
    
    parser = argparse.ArgumentParser(description='PyTorch Super Parameter Example')
    parser.add_argument('--scale_ratio', type=int, default=8, help="super resolution upscale factor")
    parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
    parser.add_argument('--n_bands', type=int, default=31)
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=456, help='random seed to use. Default=123')
    parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
    parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
    parser.add_argument('--local_rank', default=1, type=int, help='None')
    parser.add_argument('--use_distribute', type=int, default=1, help='None')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    parser.add_argument('--n_select_bands', type=int, default=3)
    parser.add_argument('--in_channels', type = int, default = 31, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 31, help = 'output channels for generator')

    #model setting
    parser.add_argument('--arch', type=str, default='SSFEINet',
                        choices=[ 
                            # the proposed method
                            'SSFEINet',
                        ])
    #Train setting
    parser.add_argument('--train_data_path', default='data/Train/', type=str,
                        help='Path of the training data')
    parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')
    parser.add_argument("--trainset_num", default=20000, type=int, 
                        help='The number of training samples of each epoch')
       
    #Test setting
    parser.add_argument('--test_data_path', default='data/Test/', type=str, 
                        help='path of the testing data')
    parser.add_argument("--testset_num", default=12, type=int, help='total number of testset')
    parser.add_argument("--test_batch_size", default=1, type=int, help='testing batch size')

    args = parser.parse_args()
    
    return args
