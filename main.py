import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import time
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='VGG16 for classification in cifar10 Training With Pytorch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                    help='yes or no to choose using warmup strategy to train')
parser.add_argument('--wp_epoch', type=int, default=5,
                    help='The upper bound of warm-up')
args = parser.parse_args()

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'lr_epoch': (75, 120)}


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


class VGG(nn.Module):
    def __init__(self, base):
        super(VGG, self).__init__()
        # self.features = nn.ModuleList(base)
        self.features = nn.Sequential(*base)
        self.classifier = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # bias: bool  默认为True
            # 线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10, bias=False)
        )

        # 初始化权重
        for m in self.modules():
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(m, nn.Conv2d):
                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(-1, 512)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        return x


# 数据获取(数据增强,归一化)
def transforms_RandomHorizontalFlip():
    # transforms.Compose(),将一系列的transforms有序组合,实现按照这些方法依次对图像操作

    # ToTensor()使图片数据转换为tensor张量,这个过程包含了归一化,图像数据从0~255压缩到0~1,这个函数必须在Normalize之前使用
    # 实现原理,即针对不同类型进行处理,原理即各值除以255,最后通过torch.from_numpy将PIL Image或者 numpy.ndarray()针对具体类型转成torch.tensor()数据类型

    # Normalize()是归一化过程,ToTensor()的作用是将图像数据转换为(0,1)之间的张量,Normalize()则使用公式(x-mean)/std
    # 将每个元素分布到(-1,1). 归一化后数据转为标准格式,

    transform_train = transforms.Compose([transforms.Pad(4),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # root:cifar-10 的根目录,data_path
    # train:True=训练集, False=测试集
    # transform:(可调用,可选)-接收PIL图像并返回转换版本的函数
    # download:true=从互联网上下载数据,并将其放在root目录下,如果数据集已经下载,就什么都不干
    train_dataset = datasets.CIFAR10(root='/data/datasets/', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='/data/datasets/', train=False, transform=transform, download=True)

    return train_dataset, test_dataset


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # 数据增强:随机翻转
    train_dataset, test_dataset = transforms_RandomHorizontalFlip()

    '''
    #Dataloader(....)
    dataset:就是pytorch已有的数据读取接口,或者自定义的数据接口的输出,该输出要么是torch.utils.data.Dataset类的对象,要么是继承自torch.utils.data.Dataset类的自定义类的对象

    batch_size:如果有50000张训练集,则相当于把训练集平均分成(50000/batch_size)份,每份batch_size张图片
    train_loader中的每个元素相当于一个分组,一个组中batch_size图片,

    shuffle:设置为True时会在每个epoch重新打乱数据(默认:False),一般在训练数据中会采用
    num_workers:这个参数必须>=0,0的话表示数据导入在主进程中进行,其他大于0的数表示通过多个进程来导入数据,可以加快数据导入速度
    drop_last:设定为True如果数据集大小不能被批量大小整除的时候,将丢到最后一个不完整的batch(默认为False)
    '''

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    net = VGG(vgg(cfg['VGG16'], 3, False))  # 这里的net就是VGG16

    if args.cuda:  # 多卡并行
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    epoch_size = len(train_dataset) // args.batch_size
    base_lr = args.lr
    criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵
    net.train()
    acc = []
    start = time.time()
    for epoch in range(150):
        train_loss = 0.0

        # 使用阶梯学习率衰减策略
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (inputs, labels) in enumerate(train_loader, 0):
            # 使用warm-up策略来调整早期的学习率
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)

            # 将数据从train_loader中读出来,一次读取的样本是32个
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels).cuda()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if iter_i % 100 == 99:
                print('[epoch: %d iteration: %5d] loss: %.3f' % (epoch + 1, iter_i + 1, train_loss / 100))
                train_loss = 0.0
        lr_1 = optimizer.param_groups[0]['lr']
        print("learn_rate:%.15f" % lr_1)
        if epoch % 5 == 4:
            print('Saving epoch %d model ...' % (epoch + 1))
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(net.state_dict(), './checkpoint4/cifar10_epoch_%d.pth' % (epoch + 1))

            # 由于训练集不需要梯度更新,于是进入测试模式
            net.eval()
            correct = 0.0
            total = 0
            with torch.no_grad():  # 训练集不需要反向传播
                print("=======================test=======================")
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = net(inputs)

                    pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
                    total += inputs.size(0)
                    correct += torch.eq(pred, labels).sum().item()

            print("Accuracy of the network on the 10000 test images:%.2f %%" % (100 * correct / total))
            print("===============================================")

            acc.append(100 * correct / total)
    print("best acc is %.2f, corresponding epoch is %d" % (max(acc), (np.argmax(acc) + 1) * 5))
    print("===============================================")
    end = time.time()
    print("time:{}".format(end - start))