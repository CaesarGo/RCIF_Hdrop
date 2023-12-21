# RCIF: Towards Robust Distributed DNN Collaborative Inference under Highly Lossy Networks

### Overview

RCIF aims to maximize the accuracy of neural network inference in DNN co-inference scenarios with data losses. So far, the main content of the project includes four folders:

- seg_cifar10:  A summary of all the experiments on CIFAR-10 dataset. The DNNs used include resnet18, resnet50 and mobilenetv2. These experiments refers to the basic project pytorch-cifar: https://github.com/kuangliu/pytorch-cifar.
- seg_det:  A summary of all the experiments on PASCAL2007+ 2012 dataset. The object detection network used is YOLOv5s. These experiments refers to the basic project YOLOv5: https://github.com/ultralytics/yolov5
- wireless_traces: part of the wireless loss trace files we obtained in the real-world wireless experiments.
- examples: For simple validation.

In the experiment, both the training and testing phases were conducted using PyTorch and a single RTX 3090 graphics card with CUDA version 11.7. The python environment used for the experiment has been recorded in the requirement.txt file. To install the related packages, run:

`pip install -r requirements.txt`

### Dataset prepare

The CIFAR-10 dataset can be automatically downloaded in PyTorch. 

For preparing the PASCAL VOC 2007+2012 dataset, please refer to the official instructions provided by the YOLOv5 project.

### Quick validation

#### For quickest validation: 

Since the checkpoints of MobileNetv2 is small, we put it directly into the supplementary material directory, so it can be run directly. After configuring the environment according to the previous steps and preparing the dataset, it can be used.  Enter directory **/example/mobilev2_cifar10**, and type following command for example (split at layer3):

`python main.py --ckpt mobile_base.pth --lose lose_0 --type nodrop `(base model, without lose)

`python main.py --ckpt mobile_drop_layer0_3.pth --lose lose_571 --type drop `(standard dropout model,  lose =57.1%)

`python main.py --ckpt mobile_hdrop_layer0_3.pth --lose lose_292 --type hdrop `(h-dropout model, lose= 29.2%)

The current code includes seven lose rate parameters: {lose_0, lose_34, lose_116, lose_292, lose_406, lose_571, lose_730}. You can choose any one of them for performance comparison testing. If you want to test other packet loss rates, you can also add data from the wireless trace file to the code.

#### Download more checkpoints to validate:

You can quickly verify the performance of RCIF using the checkpoints we provide according to the following steps:

1. Due to the size limitations of supplementary materials in the MM conference system, we will upload our checkpoint using **Google Drive**. Download our **example checkpoints** from here  https://drive.google.com/file/d/1ehKwiepTTCAWeKjERurW4BEupFP5iX2y/view?usp=sharing .  Another file all_check_point.zip contains all the checkpoints used in the experiments in the paper: https://drive.google.com/file/d/1-ZpG-4_BMqgDqNkIuZlmG9L52LsS1XEc/view?usp=sharing. But if you want to quick validation, just download the **example checkpoints**.

2. Copy the checkpoint files into the corresponding directories.  For example, place the checkpoints of ResNet-18 under /examples/res18_cifar10/, then run the following example command:

   `python main.py --ckpt resnet18_base.pth --lose lose_0 --type nodrop `

   The command line includes three parameters：

   **--ckpt:**  the name of checkpoint files.

   **--lose:**  wireless loss rate, {lose_0, lose_34, lose_116, lose_292, lose_406, lose_571, lose_730}. You can choose any one of them for performance comparison testing. If you want to test other packet loss rates, you can also add data from the wireless trace file to the code.

   **--type: **dropout type: include nodrop, drop and hdrop.

3.  Test for yolov5s, run the following example command:

   `python val.py --batch 128 --weights no_drop.pt --data VOC.yaml --img 512`

   you can use different weights to test the performance, and change the loss rate in **val.py** by adjust the **lose_index** variable. 

### Train your own model using RCIF

1. To train a CNN based on cifar10, use /seg_cifar10/pytorch-cifar-master/main.py for training. During training, you can adjust the corresponding network structure in the models.
2. To train yolov5s based on PASCAL VOC, use /seg_det/train.py for training.