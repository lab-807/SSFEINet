# S<sup>2</sup>FEINet
This is the official code for the paper 'S<sup>2</sup>FEINet: A Spatial-Spectral Feature Extraction and Interactive Network for Fusing Hyperspectral and Multispectral Images.' 

## 1.Dataset
* Download the CAVE dataset from [here](https://www1.cs.columbia.edu/CAVE/databases/multispectral/).

* Download the NTIRE 2022 dataset from [here](https://codalab.lisn.upsaclay.fr/competitions/721).

* Download the Chikusei dataset from [here](https://naotoyokoya.com/Download.html).  

* Download the Houston 2012 dataset from [here](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/).

* Download the WorldView-2 dataset from [here](https://liangjiandeng.github.io/PanCollection.html).

## 2.Environment


## 3.Training
To train the model, run the following command.

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run  --nproc_per_node=2 --master_port 295670 main.py --arch 'SSFEINet'

## 4.Testing
To experiment with saved model checkpoints, run the following command. Download the checkpoint from this link: [Baidu Netdisk](https://pan.baidu.com/s/1VlUICP-LBmPbeswIyNKeoA?pwd=kf83). Please place the trained model in the 'TrainedNet' folder.

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run  --nproc_per_node=2 --master_port 295670 test.py --arch 'SSFEINet'



