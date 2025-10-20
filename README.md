# S<sup>2</sup>FEINet
This is the official code for the paper 'S<sup>2</sup>FEINet: A Spatial-Spectral Feature Extraction and Interactive Network for Fusing Hyperspectral and Multispectral Images.' 
## Training
To train the model, run the following command.

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run  --nproc_per_node=2 --master_port 295670 main.py --arch 'SSFEINet'

## Testing
To experiment with saved model checkpoints, run the following command.Download the checkpoint from this link: [Baidu Netdisk](https://pan.baidu.com/s/1VlUICP-LBmPbeswIyNKeoA?pwd=kf83). Please place the trained model in the 'TrainedNet' folder.

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run  --nproc_per_node=2 --master_port 295670 test.py --arch 'SSFEINet'

## CAVE Dataset


