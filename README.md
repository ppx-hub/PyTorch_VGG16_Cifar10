# PyTorch_VGG16_Cifar10
转换SNN的ANN版本（即去掉bias和bn层）



### 网络结构

> VGG(
> 
>   (features): 
>   Sequential(
>     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (1): ReLU(inplace=True)
>     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (3): ReLU(inplace=True)
>     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
>     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (6): ReLU(inplace=True)
>     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (8): ReLU(inplace=True)
>     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
>     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (11): ReLU(inplace=True)
>     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (13): ReLU(inplace=True)
>     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (15): ReLU(inplace=True)
>     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
>     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (18): ReLU(inplace=True)
>     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (20): ReLU(inplace=True)
>     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (22): ReLU(inplace=True)
>     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
>     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (25): ReLU(inplace=True)
>     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (27): ReLU(inplace=True)
>     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
>     (29): ReLU(inplace=True)
>     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
>   )
>   
>   (classifier): Sequential(
>     (0): Linear(in_features=512, out_features=512, bias=False)
>     (1): ReLU(inplace=True)
>     (2): Dropout(p=0.5, inplace=False)
>     (3): Linear(in_features=512, out_features=256, bias=False)
>     (4): ReLU(inplace=True)
>     (5): Dropout(p=0.5, inplace=False)
>     (6): Linear(in_features=256, out_features=10, bias=False)
>   )
> )



### 硬件

2块A100 PCIe 40GB，150个epoch所需时间：3472.9s



### Top-1 Accuracy

Best_acc: 91.93%(在epoch为130时)，即error为8.1%

<img src="https://gitee.com/Hexaaaaaa/blogimage/raw/master/img/20220212124834.png" style="zoom:50%;" />

