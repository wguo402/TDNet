# TDNet: Triplet Diffusion Network for 3D Point Clouds Object Tracking

## Demo

![TDNet](https://github.com/wguo402/TDNet/blob/main/demo.gif)

The above is our fragment display, and the more complete object tracking process is shown in [demo](https://www.bilibili.com/video/BV1Ct421V7Mz/?spm_id_from=333.999.0.0&vd_source=94575dffc53c970af321acb0619d64b4).

## Introduction

To address some challenges such as position changes caused by occlusions, semantic information deficiency due to the sparsity of the point cloudin 3D point cloud tracking, we put forward a novel tracking method by a Triplet Diffusion Network(TDNet), which is primarily composed of diffusion enhancement, template updating, and multi-scale feature matching. 

A comprehensive evaluation of the proposed approach on various 3D point cloud benchmark sequences on KITTI, nuScenes, and Waymo has been performed. 

This code is only to give paper reviewers a verification and academic research. After the paper is accepted, we will polish and optimize the code.


## Environment settings
* Create an environment for TDNet
```
conda create -n TDNet python=3.7
conda activate TDNet
```

* Install pytorch and torchvision
```
conda install pytorch==1.7.0 torchvision==0.5.0 cudatoolkit=10.0
```

* Install dependencies.
```
pip install -r requirements.txt
```

## Data preparation
### [KITTI dataset](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)
* Download the [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Unzip the downloaded files and place them under the same parent folder.

### [nuScenes dataset](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)
* Download the Full dataset (v1.0) from [nuScenes](https://www.nuscenes.org/).
  
    Note that base on the offical code [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit), we modify and use it to convert nuScenes format to KITTI format. It requires metadata from nuScenes-lidarseg. Thus, you should replace *category.json* and *lidarseg.json* in the Full dataset (v1.0). We provide these two json files in the nuscenes_json folder.

    Note that the parameter of "split" should be "train_track" or "val". In our paper, we use the model trained on the KITTI dataset to evaluate the generalization of the model on the nuScenes dataset.
	
### [Waymo open dataset](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf)
* We follow the benchmark created by [LiDAR-SOT](https://github.com/TuSimple/LiDAR_SOT) based on the waymo open dataset. You can download and process the waymo dataset as guided by [their code](https://github.com/TuSimple/LiDAR_SOT), and use our code to test model performance on this benchmark.
* The benchmark they built have many things that we don't use, but the following processing results are necessary:

**Node**: After you get the dataset, please modify the path variable ```data_dir&val_data_dir``` about the dataset under configuration file ```./utils/options```.

## Evaluation

Train a model:

```
cd diffusion

python train_gen.py --which_dataset KITTI/NUSCENES --category_name category_name
```

```
python main.py --which_dataset KITTI/NUSCENES --category_name category_name
```

Test a model:
```
python main_test.py --which_dataset KITTI/NUSCENES/WAYMO --category_name category_name --train_test test
```
For more preset parameters or command debugging parameters, please refer to the relevant code and change it according to your needs.

**Recommendations**: 
- We have provided some pre-trained models under ```./results``` folder, you can use and test them directly.  
- Since both kitti and waymo are datasets constructed from 64-line LiDAR, nuScenes is a 32-line LiDAR. We recommend you: train your model on KITTI and verify the generalization ability of your model on waymo. Train on nuScenes or simply skip this dataset. We do not recommend that you verify the generalization ability of your model on nuScenes. 

## Citation

```
@inproceedings{hui2022stnet,
  title={3D Siamese Transformer Network for Single Object Tracking on Point Clouds},
  author={Hui, Le and Wang, Lingpeng and Tang, Linghua and Lan, Kaihao and Xie, Jin and Yang, Jian},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{luo2021diffusion,
  author = {Luo, Shitong and Hu, Wei},
  title = {Diffusion Probabilistic Models for 3D Point Cloud Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021}
}
```

## Reference

- Thank Hui for his implementation of [STNet](https://arxiv.org/pdf/2207.11995.pdf). 

- Thank Luo for his implementation of [DPM](https://arxiv.org/pdf/2207.11995.pdf). 


