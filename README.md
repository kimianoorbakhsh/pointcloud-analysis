# pointcloud-analysis
Codes for the Point Cloud Curvature and Saliency Analysis Project. 
- Image Processing Lab, Sharif University of Technology

Note: In progress, new implementations regarding to this project will be added soon.

# Structure
- `src/pointnet`: Contains PyTorch implementation of the [Pointnet](https://arxiv.org/abs/1612.00593) model and all the related utility functions to train and evaluate it on the ModelNet data.
- `src/dgcnn`:  Contains PyTorch implementation of the [DGCNN](https://arxiv.org/abs/1801.07829) model and all the related utility functions to train and evaluate it on the ModelNet data.
- `src/saliency_map.py`: Contains the PyTorch implementation of the [Point Cloud Saliency Maps.](https://arxiv.org/abs/1812.01687)
- `src/deep_fool.py`: Contains code for the [DeepFool method](https://arxiv.org/pdf/1511.04599.pdf), a simple and accurate method to fool deep neural networks (Pytorch & PointCloud Implementation).
## Models

Model checkpoints are available [here](https://github.com/kimianoorbakhsh/pointcloud-analysis/tree/main/models). 

## Datacets

ModelNet Dataset from [Priceton ModelNet](https://modelnet.cs.princeton.edu/).

## Citation
Please cite us if you use our work in your research.

## References:
- https://github.com/aryanmikaeili/Pointnet-CW-attack
- https://github.com/tianzheng4/PointCloud-Saliency-Maps
