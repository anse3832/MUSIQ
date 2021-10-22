# MUSIQ: Multi-Scale Image Quality Transformer
Unofficial pytorch implementation of the paper "MUSIQ: Multi-Scale Image Quality Transformer"
(paper link: https://arxiv.org/abs/2108.05997)

This code doesn't exactly match what the paper describes.
- It only works on the KonIQ-10k dataset. Or it works on the database which resolution is 1024(witdh) x 768(height).
- Instead of using 5-layer Resnet as a backbone network, we use ResNet50 pretrained on ImageNet database.
- We need to implement Earth Mover Distance (EMD) loss to train on other databases.
- We additionally use ranking loss to improve the performance (we will upload the training code including ranking loss later)

The environmental settings are described below. (I cannot gaurantee if it works on other environments)
- Pytorch=1.7.1 (with cuda 11.0)
- einops=0.3.0
- numpy=1.18.3
- cv2=4.2.0
- scipy=1.4.1
- json=2.0.9
- tqdm=4.45.0

# Train & Validation
First, you need to download weights of ResNet50 pretrained on ImageNet database.
- Downlod the weights from this website (https://download.pytorch.org/models/resnet50-0676ba61.pth)
- rename the .pth file as "resnet50.pth" and put it in the "model" folder

Second, you need to download the KonIQ-10k dataset.
- Download the database from this website (http://database.mmsp-kn.de/koniq-10k-database.html)
- set the database path in "train.py" (It is represented as "db_path" in "train.py")
- Please check "koniq-10k.txt" is in "IQA_list" folder
- "koniq-10k.txt" file includes [scene number / image name / ground truth score] information

After those settings, you can run the train & validation code by running "train.py"
- python3 train.py (execution code)
- This code works on single GPU. If you want to train this code in muti-gpu, you need to change this code
- Options are all included in "train.py". So you should change the variable "config" in "train.py"
![image](https://user-images.githubusercontent.com/77471764/138195607-cf7165a1-dd64-4031-b1ab-872012f7046a.png)

Belows are the validation performance on KonIQ-10k database (I'm still training the code, so the results will be updated later)
- SRCC: 0.9023 / PLCC: 0.9232 (after training 105 epochs)
- If the codes are implemented exactly the same as the paper, the performance can be further improved

# Inference
First, you need to specify variables in "inference.py"
- dirname: root folder of test images
- checkpoint: checkpoint file (trained on KonIQ-10k dataset)
- result_score_txt: inference score will be saved on this txt file
![image](https://user-images.githubusercontent.com/77471764/138195041-3176224f-6ab6-42b1-aa61-f9ec8a1ffa96.png)

After those settings, you can run the inference code by running "inference.py"
- python3 inference.py (execution code)

# Acknolwdgements
We refer to the following website to implement the transformer (https://paul-hyun.github.io/transformer-01/)
