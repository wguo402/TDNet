import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import  math
from utils.loss.losses import RegL1Loss, FocalLoss
from utils.loss.PCLosses import ChamferLoss
from diffusion.utils.misc import *
from diffusion.utils.dataset import *
from diffusion.utils.data import *
from diffusion.models.vae_gaussian import *
from diffusion.models.vae_flow import *
from diffusion.evaluation import *


from datasets.get_stnet_db import get_dataset
from modules.stnet import STNet_Tracking
from utils.show_line import print_info
from trainers.trainer import train_model, valid_model
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


def train_tracking(opts):
    ## Init
    print_info(opts.ncols, 'Start')
    set_seed(opts.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt1', type=str,
                        default='')
    parser.add_argument('--ckpt2', type=str,
                        default='')
    parser.add_argument('--categories', type=str_list, default=['Pedes'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    # Datasets and loaders
    parser.add_argument('--batch_size', type=int, default=64)
    # Sampling
    parser.add_argument('--sample_num_points1', type=int, default=512)
    parser.add_argument('--sample_num_points2', type=int, default=512)
    parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
    parser.add_argument('--seed', type=int, default=9988)
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger('test', save_dir)
    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))

    ckpt1 = torch.load(args.ckpt1)
    ckpt2 = torch.load(args.ckpt2)
    seed_all(args.seed)


    ## Define dataset
    print_info(opts.ncols, 'Define dataset')
    train_loader, train_db, train_iter = get_dataset(opts, partition="Train", shuffle=True)
    valid_loader, valid_db, val_iter = get_dataset(opts, partition="Valid", shuffle=False)

    opts.voxel_size = torch.from_numpy(train_db.voxel_size.copy()).float()
    opts.voxel_area = train_db.voxel_grid_size
    opts.scene_ground = torch.from_numpy(train_db.scene_ground.copy()).float()
    opts.min_img_coord = torch.from_numpy(train_db.min_img_coord.copy()).float()
    opts.xy_size = torch.from_numpy(train_db.xy_size.copy()).float()

    ##first diffusion model

    if ckpt1['args'].model == 'gaussian':
        model1 = GaussianVAE(ckpt1['args']).to(args.device)
        model2 = GaussianVAE(ckpt2['args']).to(args.device)
    elif ckpt1['args'].model == 'flow':
        model1 = FlowVAE(ckpt1['args']).to(args.device)
        model2 = FlowVAE(ckpt2['args']).to(args.device)
    logger.info(repr(model1))
    logger.info(repr(model2))
    model1.load_state_dict(ckpt1['state_dict'])
    model2.load_state_dict(ckpt2['state_dict'])

    # Reference Point Clouds
    ref_pcs_tem=[]
    ref_pcs_sea=[]

    # Generate Point Clouds
    gen_pcs_tem = []
    gen_pcs_sea = []
    for i in tqdm(range(0, math.ceil(len(train_db) / args.batch_size)), 'Generate'):
        with torch.no_grad():
            if i < int(len(train_db) / args.batch_size)  or (len(train_db) % args.batch_size)==0:
                z1 = torch.randn([args.batch_size, ckpt1['args'].latent_dim]).to(args.device)
                z2 = torch.randn([args.batch_size, ckpt2['args'].latent_dim]).to(args.device)
                x_tem = model1.sample(z1, args.sample_num_points1, flexibility=ckpt1['args'].flexibility)
                x_sea = model2.sample(z2, args.sample_num_points2, flexibility=ckpt2['args'].flexibility)
                gen_pcs_tem.append(x_tem.detach().cpu())
                gen_pcs_sea.append(x_sea.detach().cpu())
            else:
                z1 = torch.randn([len(train_db) % args.batch_size, ckpt1['args'].latent_dim]).to(args.device)
                z2 = torch.randn([len(train_db) % args.batch_size, ckpt2['args'].latent_dim]).to(args.device)
                x_tem = model1.sample(z1, args.sample_num_points1, flexibility=ckpt1['args'].flexibility)
                x_sea = model2.sample(z2, args.sample_num_points2, flexibility=ckpt2['args'].flexibility)
                gen_pcs_tem.append(x_tem.detach().cpu())
                gen_pcs_sea.append(x_sea.detach().cpu())

    ## Define model
    print_info(opts.ncols, 'Define model')
    model = STNet_Tracking(opts)
    if (opts.n_gpus > 1) and (opts.n_gpus >= torch.cuda.device_count()):
        model = torch.nn.DataParallel(model, range(opts.n_gpus))
    model = model.to(opts.device)

    ## Define optim & scheduler
    print_info(opts.ncols, 'Define optimizer & scheduler')
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, betas=(0.9, 0.999))

    if opts.which_dataset.upper() == "NUSCENES":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    ## Define loss
    print_info(opts.ncols, 'Define loss')
    criternions = {
        'hm': FocalLoss().to(opts.device),
        'loc': RegL1Loss().to(opts.device),
        'z_axis': RegL1Loss().to(opts.device),
    }

    ## Training
    print_info(opts.ncols, 'Start training!')

    best_loss = 9e99
    for epoch in range(1,opts.n_epoches + 1):
        print('Epoch', str(epoch), 'is training:')
        train_loss = train_model(opts, model, train_loader, optimizer, criternions, epoch,gen_pcs_tem,gen_pcs_sea)
        # save current epoch state_dict
        torch.save(model.state_dict(), os.path.join(opts.results_dir, "netR_" + str(epoch) + ".pth"))
        # update scheduler
        scheduler.step(epoch)

