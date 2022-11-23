# Deep Learning Mini-Project: Modifying ResNet on Image Classification

## Reference: used https://github.com/kuangliu/pytorch-cifar as reference

<br>

## Authors
- Jianglong He
- Kaiyun Kang
- Wenxin Li

<br>

## Description
In the project, we aim to obtain high test accuracy on CIFAR-10 image classification dataset with truncated version of ResNet-18 down to less than 5 million parameters. We conducted extensive experiment to figure out the best set of hyperparameters and model architeture.

<br>

## Best Model
### Number of Parameters: 4,779,530
### Testset Accuracy: 92.49

### Model Architecture:
- number of residual layers: 4
- residual blocks per residual layers: [3, 4, 4, 3]
- average pool of size 4
- linear layers: 2 
    - shape of: (256, 128), (128, 10)

### Data Augmentations:
- Normalization:
    - mean: (0.4914, 0.4822, 0.4465)
    - std: (0.2023, 0.1994, 0.2010)
- Random Crop:
    - size: 32
    - padding: 4
- Random Horizontal Flip:
    - p: 0.5
- Random Rotation:
    - degrees: 5

### Hyperparameters:
- total epochs: 200
- train batch size: 128
- learning rate: 0.001
- weight decay: 5e-4
- optimizer: AdamW


<br>

## Install Dependencies
```
pip3 install -r requirements.txt
```

<br>

## Training Procedures
there are three main files that we modify to tune our model. `train.py`, `config.py`, and `model/resnet.py`. Each leaf folder inside `outputs/` contains `train.py`, `config.py`, `model/resnet.py` and a `txt` file storing the training log that is generated when we conduct the training.

To trigger training: 
1. first get these `train.py`, `config.py`, and `model/resnet.py` files from `outputs/` directory
2. replace old `train.py`, `config.py`, and `model/resnet.py` with the new ones you get from step 1
3. to trigger the training run following command:
    ```
    python3 train.py
    ```
4. trained model and log file will be generated to the path specified by `output_dir` in your `config.py` file.

<br>

## Reproduce Experiments

### Best Model Experiment
| Method      | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
| Best Model|4,779,530|200|99.63556|93.0400|92.49000|[folder](outputs/model_arch/l4-3443-lr0.001-linear128-epoch200/config.py)|

<br>

### Data Augmentation Experiments
| Method      | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
| Baseline|2.77M|18|86.58|78.7|78.29|[folder](outputs/baseline/config.py)|
| Normalization|2.77M|19|87.22|79.20|79.58|[folder](outputs/data_aug/norm/config.py)|
| Normalization <br> Random Crop|2.77M|16|83.33|79.54|78.32|[folder](outputs/data_aug/norm_crop/config.py)|
| Normalization <br> Random Crop <br> Random Flip|2.77M|14|82.16|76.38|76.49|[folder](outputs/data_aug/norm_crop_flip/config.py)|
| Normalization <br> Random Crop <br> Random Flip <br> Random Rotation|2.77M|28|81.64|81.14|80.76|[folder](outputs/data_aug/norm_crop_flip_rotate5/config.py)|

<br>

### Learning Rate and Weight Decay Experiments
| Method <br> LearningRate/WeightDecay      | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
| 0.01 / 0.01   | 2.77M | 26 | 92.95 | 89.12 | 88.72 |[folder](outputs/lr_wd/wd0.01/config.py)|
| 0.01 / 0.003  | 2.77M | 38 | 96.35 | 90.70 | 89.88 |[folder](outputs/lr_wd/wd0.003/config.py)|
| 0.01 / 5e-4   | 2.77M | 34 | 96.62 | 90.56 | 90.32 |[folder](outputs/lr_wd/wd5e-4/config.py)|
| 0.001 / 0.003 | 2.77M | 27 | 95.25 | 90.08 | 89.06 |[folder](outputs/lr_wd/wd0.003_lr0.001/config.py)|
| 0.001 / 5e-4  | 2.77M | 30 | 95.86 | 90.42 | 89.75 |[folder](outputs/lr_wd/wd5e-4_lr0.001/config.py)|


<br>

### Training Batch Size Experiments
| Method <br> LearningRate/WeightDecay      | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
| 32             | 2.77M | 23 | 93.08 | 89.06 | 88.81 | [folder](outputs/batch_size/bs32/config.py) |
| 64             | 2.77M | 29 | 95.45 | 89.98 | 89.54 |  [folder](outputs/batch_size/bs64/config.py)|
| 128 (baseline) | 2.77M | 34 | 96.62 | 90.56 | 90.32 |[folder](outputs/lr_wd/wd5e-4/config.py)|
| 256            | 2.77M | 32 | 96.31 | 90.00 | 89.21 | [folder](outputs/batch_size/bs256/config.py) |


<br>

### Optimizer Experiments
| Method <br> Adam:LR/WD <br> SGD:LR/WD/Momentum      | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
|AdamW (0.01, 5e-4) | 2.77M | 34 | 96.62 | 90.56 | 90.32 | [folder](outputs/lr_wd/wd5e-4/config.py) |
| SGD (0.01, 5e-4, 0.9)        | 2.77M | 28 | 93.66 | 89.34 | 89.66 | [folder](outputs/optimizer/sgd_wd5e-4_lr0.01/config.py) |
| SGD (0.001, 5e-4, 0.9)       | 2.77M | 28 | 89.01 | 84.84 | 84.59 | [folder](outputs/optimizer/sgd_wd5e-4_lr0.001/config.py) |
| SGD (0.01, 0.01, 0.9)        | 2.77M | 13 | 80.59 | 75.66 | 75.30 | [folder](outputs/optimizer/sgd_wd0.01_lr0.01/config.py) |
| SGD (0.01, 0.001, 0.9)       | 2.77M | 18 | 90.14 | 85.94 | 85.57 |[folder](outputs/optimizer/sgd_wd0.001_lr0.01/config.py)  |
| ADAM (0.01 / 5e-4)                      | 2.77M | 9  | 64.22 | 60.54 | 58.64 |  [folder](outputs/optimizer/adam_wd5e-4_lr0.01/config.py)|
| ADAM (0.001 / 5e-4)                     | 2.77M | 37 | 89.97 | 87.88 | 87.88 |  [folder](outputs/optimizer/adam_wd5e-4_lr0.001/config.py)|
| ADAM (0.001 / 0.01)                     | 2.77M | 26 | 76.92 | 74.30 | 73.69 | [folder](outputs/optimizer/adam_wd0.01_lr0.001/config.py) |

<br>

### Residual Blocks (Bi) Experiments
| Method    | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
| #Residual blocks in Residual i: \[2, 3, 3\]      | 4,253,770 | 28 | 96.91778 | 87.86 | 88.10 | [folder](outputs/num_blocks/block-233/config.py) |
| #Residual blocks in Residual Layer i:\[3,3,3\] | 4,327,754 | 27 | 96.56444 | 87.80 | 87.54 | [folder](outputs/num_blocks/block-333/config.py) |
| #Residual blocks in Residual i: \[2, 4, 3\]      | 4,549,194 | 30 | 95.93333 | 89.94 | 89.18 | [folder](outputs/num_blocks/block-243/config.py) |
|#Residual blocks in Residual i: \[3,5, 3\]       | 4,918,602 | 29 | 96.25778 | 89.24 | 89.53 | [folder](outputs/num_blocks/block-353/config.py) |


<br>

### Model Architecture with Channels (Ci) Experiments
- Ci: the number of channels in Residual Layer i

| Method    |Number of Residual Layers | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      | :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
|Ci: \[64, 128, 256\]| N: # Residual Layers: 3 | 2777674 | 33 | 95.62222 | 89.66000 | 88.46000 | [folder](outputs/model_arch/l3-pool8/config.py) |
|Ci: \[32, 64, 128, 256\]| N: # Residual Layers: 4| 2809866 | 32 | 93.00444 | 89.3200 | 88.91000 | [folder](outputs/model_arch/l4-pool4/config.py) |
|Ci: \[16, 32, 64, 128, 256\] |N: # Residual Layers: 5| 2811818 | 36 | 90.20444 | 87.5000 | 87.17000 | [folder](outputs/model_arch/l5-pool2/config.py) |


<br>

### Conv.Kernel Size (Fi) in Residual Layer i Experiments
| Method Kernel Sizes  | Number of Blocks | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
|5-1515-1511515-15115 <br>  |\# of blocks: \[2,3,2\]| 4911178 | 24 | 94.68667 | 87.26 | 87.60 | [folder](outputs/num_kernels/kernel-15/config.py)  |
|Bottleneck: 131|\# of blocks: \[3,3,3\]                      | 4914250 | 24 | 90.69111 | 85.66 | 87.12 | [folder](outputs/num_kernels/kernel-131/config.py) |
| 5\*5|\# of blocks: \[1,31\] | 4964170 | 20 | 93.89778 | 88.34 | 87.49 | [folder](outputs/num_kernels/kernel-5/config.py)   |


<br>

### Average Pool Size (P) Experiments
| Method Pool Size Sizes  | Number of Blocks | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
| 4 | # of block \[2, 4, 3\]       | 4556874 | 24 | 93.47333 | 89.20 | 88.32 | [folder](outputs/average-pool-size/pool-4/pool4-block243/config.py) |
| 4 | # of block \[3, 5, 3\]       | 4926282 | 40 | 96.77778 | 89.98 | 89.52 | [folder](outputs/average-pool-size/pool-4/pool4-block353/config.py) |
| 2 | # of block \[3, 5, 3\]| 4957002 | 40 | 94.40667 | 88.88 | 88.02 | [folder](outputs/average-pool-size/pool-2/config.py)               |


<br>

### Adding More Layers Experiments
| Method    | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
| Prev Best | 2.77M | 34 | 96.62 | 90.56 | 90.32 |[folder](outputs/lr_wd/wd5e-4/config.py)| 
| Prev Best <br> Conv2d\_BN (256, 512) | 3.96M | 29                      | 95.33 | 91.10 | 89.89 |[folder](outputs/extra_layers/extraConvBn/config.py)| 
|Conv2d\_BN (256, 512) <br> Linear1 (512, 256) <br> Linear2 (256, 128) <br> Linear (128, 10)| 4.12M | 25                      | 92.38 | 89.22 | 87.65 |[folder](outputs/extra_layers/extraConvBn/config.py) |
|Conv2d\_BN (256, 512) <br> Dropout(0.2) <br> Linear1 (512, 256) <br> Dropout(0.2) <br> Linear2 (256, 128) <br> Linear (128, 10) | 4.12M | 37 | 94.71 | 90.38 | 89.51 |[folder](outputs/extra_layers/extraConvBn_3fcs_2dropouts0.2/config.py)|
| Conv2d\_BN (256, 512) <br> Dropout(0.4) <br> Linear1 (512, 256) <br> Dropout(0.4) <br> Linear2 (256, 128) <br> Linear (128, 10) | 4.12M | 25 | 91.42 | 89.4  | 88.83 |[folder](outputs/extra_layers/extraConvBn_3fcs_2dropout0.4/config.py)|
|Conv2d\_BN (256, 512) <br> Dropout(0.2) <br> Linear1 (512, 256) <br> Dropout(0.2) <br> Linear2 (256, 128) <br> Dropout(0.2) <br> Linear (128, 10) | 4.12M | 39|  94.64 | 91.30 | 90.26 |[folder](outputs/extra_layers/extraConvBn_3fcs_3dropouts0.2/config.py)|
|Conv2d\_BN (256, 512) <br> Dropout(0.4) <br> Linear1 (512, 256) <br> Dropout(0.4) <br> Linear2 (256, 128) <br> Dropout(0.4) <br> Linear (128, 10) | 4.12M | 25 | 89.39 | 87.66 | 86.50 |[folder](outputs/extra_layers/extraConvBn_3fcs_3dropout0.4/config.py)|
|Conv2d\_BN (256, 512) <br> Dropout(0.2) <br> Linear1 (512, 10)| 3.96M | 28  | 94.88 | 90.26 | 89.35 |[folder](outputs/extra_layers/extraConvBn_1fcs_1dropout0.2/config.py)|
| 1st Layer ResNet (C=64) <br> 2st Layer ResNet (C=128) <br> 3st Layer ResNet (C=326) <br> Dropout(0.2) <br> Linear1 (1304, 512)<br> Dropout(0.2) <br> Linear1 (512, 256) <br> Dropout(0.2)<br> Linear1 (256, 10) | 4.77M | 37 | 92.18 | 87.06 | 87.06 |[folder](outputs/extra_layers/extraConvBn_3fcs_3dropouts0.2_changedChannels/config.py)|



<br>

### Run for Longer Epochs (200) Experiments
| Method    | # of parameters | # of epoch    |  Train Accuracy |  Validation Accuracy | Test Accuracy | Folder Name |
| :----:      |    :----:   |   :----:   |   :----:   |   :----:   |   :----:   |   :----:   |
|N: # Residual Layers: 4 <br> Bi:  #Residual blocks in Residual i: \[3, 4, 4, 3\] <br> learning rate 0.001 <br> Linear: 128 <br> Epoch 200                          | 4779530 | 200 | 99.63556 | 93.0400  | 92.49000 | [folder](outputs/model_arch/l4-3443-lr0.001-linear128-epoch200/config.py)|
|N: # Residual Layers: 4 <br> Bi:  #Residual blocks in Residual i: \[3, 4, 4, 3\] <br> learning rate 0.001 <br> Linear: 128+64 <br> Epoch 200                       | 4787146 | 200 | 99.61556 | 92.40000 | 92.11000 |[folder](outputs/model_arch/l4-3443-lr0.001-linear128+64-epoch200/config.py)|
|N: # Residual Layers: 4 <br> Bi:  #Residual blocks in Residual i: \[3, 4, 4, 3\] <br> learning rate 0.001 <br> Linear: 128+64 <br> Dropout: 0.2 <br> Epoch 200 | 4787146 | 200 | 99.56889 | 92.36000 | 92.19000 | [folder](outputs/model_arch/l4-3443-lr0.001-linear128+64-dropout0.2-epoch200/config.py)|
|N: # Residual Layers: 4 <br> Bi:  #Residual blocks in Residual i: \[3, 4, 4, 3\] <br> learning rate 0.001 <br> Linear: 128+64 <br> Dropout: 0.5 <br> Epoch 200 | 4787146 | 200 | 99.57556 | 92.3000  | 92.08000 | [folder](outputs/model_arch/l4-3443-lr0.001-linear128+64-dropout0.5-epoch200/config.py)|
