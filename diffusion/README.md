## Installation

**[Option 1]** Install via conda environment YAML file (**CUDA 10.1**).

```bash
# Create the environment
conda env create -f env.yml
# Activate the environment
conda activate dpm-pc-gen
```

**[Option 2]** Or you may setup the environment manually (**If you are using GPUs that only work with CUDA 11 or greater**).

Our model only depends on the following commonly used packages, all of which can be installed via conda.

| Package      | Version                          |
| ------------ | -------------------------------- |
| PyTorch      | ≥ 1.6.0                          |
| h5py         | *not specified* (we used 4.61.1) |
| tqdm         | *not specified*                  |
| tensorboard  | *not specified* (we used 2.5.0)  |
| numpy        | *not specified* (we used 1.20.2) |
| scipy        | *not specified* (we used 1.6.2)  |
| scikit-learn | *not specified* (we used 0.24.2) |

## Training

```bash
# Train a generator
python train_gen.py
```

You may specify the value of arguments. Please find the available arguments in the script. 

Note that `--categories` can take `all` (use all the categories in the dataset), `airplane`, `chair` (use a single category), or `airplane,chair` (use multiple categories, separated by commas).

### Notes on the Metrics

Note that the metrics computed during the validation stage in the training script (`train_gen.py`, `train_ae.py`) are not comparable to the metrics reported by the test scripts (`test_gen.py`, `test_ae.py`). ***If you train your own models, please evaluate them using the test scripts***. The differences include:
1. The scale of Chamfer distance in the training script is different. In the test script, we renormalize the bounding boxes of all the point clouds before calculating the metrics (Line 100, `test_gen.py`). However, in the validation stage of training, we do not renormalize the point clouds.
2. During the validation stage of training, we only use a subset of the validation set (400 point clouds) to compute the metrics and generates only 400 point clouds (controlled by the `--test_size` parameter). Limiting the number to 400 is for saving time. However, the actual size of the `airplane` validation set is 607, larger than 400. Less point clouds mean that it is less likely to find similar point clouds in the validation set for a generated point cloud. Hence, it would lead to a worse Minimum-Matching-Distance (MMD) score even if we renormalize the shapes during the validation stage in the training script.

## Citation

```
@inproceedings{luo2021diffusion,
  author = {Luo, Shitong and Hu, Wei},
  title = {Diffusion Probabilistic Models for 3D Point Cloud Generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021}
}
```
