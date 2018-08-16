# 雪浪制造AI挑战赛—视觉计算辅助良品检测

*初赛（7月10日-8月3日，UTC+8）方案*



## 任务

布匹疵点检验。
初赛只要求做“正常”与“有瑕疵”的二分类。

## 目录结构

```
.
├── code
├── data
│   ├── xuelang_round1_test_a_20180709
│   ├── xuelang_round1_test_b
│   ├── xuelang_round1_train_part1_20180628
│   ├── xuelang_round1_train_part2_20180705
│   └── xuelang_round1_train_part3_20180709
└── submit
```

## 方案

原始图像resize后用预训练模型训练。预测时做test time augmentation（1原始+4随机）。共使用两种模型：

1. ResNet34
2. VGG16

将结果ensemble后提交。

## 运行

下载、解压文件后，执行

```
main.sh
```

## 运行环境

### 操作系统
```
Ubuntu 16.04
```

### 工具
```
CUDA Toolkit 9.0 
cuDNN v7.1.4
Python 3.6.2
```

### Python packages
```
numpy==1.14.5
pandas==0.23.3
Pillow-SIMD==5.1.1.post0
scikit-learn==0.19.2
scipy==1.1.0
torch==0.4.1
torchvision==0.2.1
tqdm==4.23.4
```