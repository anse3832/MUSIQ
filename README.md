# MUSIQ
Unofficial pytorch implementation of the paper "MUSIQ: Multi-Scale Image Quality Transformer"

This code doesn't exactly match what the paper describes.
- It only works on the KonIQ-10k dataset. Or it works on the database which resolution is 1024(witdh) x 768(height).
- Instead of using 5-layer Resnet, we use ResNet50 pretrained on ImageNet database.
- We need to implement Earth Mover Distance (EMD) loss to train on other databases.

The environmental settings are described below. (I cannot gaurantee if it works on other environments)
- Pytorch=1.7.1 (with cuda 11.0)
- einops=0.3.0
- numpy=1.18.3
- cv2=4.2.0
- scipy=1.4.1
- json=2.0.9
- tqdm=4.45.0

