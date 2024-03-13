# alraedy
import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
#from ..utils.options import opts

import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from utils.options import opts # these are in same folder as module under test!
else:
    from ..utils.options import opts


from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
from datasets.get_stnet_db import get_dataset

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/home/lqg/lzj/dataset/shapenetcore_partanno_segmentation_benchmark_v0/shapenetcore_partanno_segmentation_benchmark_v0/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=10)#float('inf')
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=30*THOUSAND)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default=None)

#----------------------------------------------------------------------------------------------------
#--------------------------------------------added---------------------------------------------------
#----------------------------------------------------------------------------------------------------
parser.add_argument('--which_dataset', type=str, default='KITTI',  help='datasets: KITT,NUSCENES,WAYMO')
parser.add_argument('--category_name', type=str, default='Cyclist',  \
    help='KITTI:Car/Pedestrian/Van/Cyclist; nuScenes:car/pedestrian/truck/bicycle; waymo:vehicle/pedestrian/cyclist')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--n_workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--n_epoches', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=2, help='# GPUs')
parser.add_argument('--train_test', type=str, default='train', help='train or test')
parser.add_argument('--model_epoch', type=int, default=30, help='which epoch model to test')
parser.add_argument('--visual', type=bool, default=False, help='save data for visualization')


args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    #log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    log_dir1 = get_new_log_dir(args.log_root, prefix='Tem_GEN_', postfix='_' + args.tag if args.tag is not None else '')
    log_dir2 = get_new_log_dir(args.log_root, prefix='Sea_GEN_', postfix='_' + args.tag if args.tag is not None else '')
    #logger = get_logger('train', log_dir)
    logger1 = get_logger('train', log_dir1)
    logger2 = get_logger('train', log_dir2)
    #writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    writer1 = torch.utils.tensorboard.SummaryWriter(log_dir1)
    writer2 = torch.utils.tensorboard.SummaryWriter(log_dir2)
    #ckpt_mgr = CheckpointManager(log_dir)
    ckpt_mgr1 = CheckpointManager(log_dir1)
    ckpt_mgr2 = CheckpointManager(log_dir2)
    # log_hyperparams(writer, args)
    log_hyperparams(writer1, args)
    log_hyperparams(writer2, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
#logger.info(args)
logger1.info(args)
logger2.info(args)

def init_opts(opts, manual_opts):
    opts.which_dataset = manual_opts.which_dataset.upper()

    if opts.which_dataset.upper() not in ['KITTI', 'NUSCENES', 'WAYMO']:
        raise ValueError("Please use command '--which_dataset kitti/nuscenes/waymo' to select datasets we support.")

    opts.batch_size = manual_opts.batch_size
    opts.n_workers = manual_opts.n_workers
    opts.n_epoches = manual_opts.n_epoches
    opts.n_gpus = manual_opts.n_gpus
    opts.train_test = manual_opts.train_test
    opts.visual = manual_opts.visual

    opts.db = opts.db[opts.which_dataset]
    if opts.which_dataset.upper() == 'KITTI' and manual_opts.category_name not in ['Car', 'Pedestrian', 'Van',
                                                                                   'Cyclist']:
        raise ValueError(
            "Please enter the correct species name supported by the KITTI dataset (Car/Pedestrian/Van/Cyclist).")
    if opts.which_dataset.upper() == 'NUSCENES' and manual_opts.category_name not in ['car', 'pedestrian', 'truck',
                                                                                      'bicycle']:
        raise ValueError(
            "Please enter the correct species name supported by the nuScenes dataset (car/pedestrian/truck/bicycle).")
    if opts.which_dataset.upper() == 'WAYMO' and manual_opts.category_name not in ['vehicle', 'pedestrian', 'cyclist']:
        raise ValueError(
            "Please enter the correct species name supported by the waymo open dataset (vehicle/pedestrian/cyclist).")
    opts.db.category_name = manual_opts.category_name

    # note that: we only use waymo oepn dataset to test the generalization ability of the kitti model
    # KITTI/WAYMO ==> kitti, NUSCENES ==> nuscenes
    # WAYMO.vehicle/pedestrian/cyclist ==> KITTI.Car/Pedestrian/Cyclist
    opts.rp_which_dataset = 'nuscenes' if opts.which_dataset.upper() == 'NUSCENES' else 'kitti'
    opts.rp_category = 'Car' if (
                opts.which_dataset.upper() == 'WAYMO' and opts.db.category_name == 'vehicle') else opts.db.category_name

    opts.data_save_path = os.path.join('/home/lqg/lzj/project2/TDNet-main/results/',
                                       ('tiny' if opts.use_tiny else 'full'), opts.rp_which_dataset)
    opts.results_dir = "./results/%s_%s" % (opts.rp_which_dataset.lower(), opts.rp_category.lower())

    if opts.train_test == 'train':
        opts.mode = True
        os.makedirs(opts.results_dir, exist_ok=True)
        os.makedirs(opts.data_save_path, exist_ok=True)
    elif opts.train_test == 'test':
        opts.mode = False

    opts.model_path = "%s/netR_%d.pth" % (opts.results_dir, manual_opts.model_epoch)

    return opts

opts = init_opts(opts, args)


train_loader, train_db,train_iter = get_dataset(opts, partition="Train", shuffle=True)
valid_loader, valid_db,val_iter = get_dataset(opts, partition="Valid", shuffle=False)

opts.voxel_size = torch.from_numpy(train_db.voxel_size.copy()).float()
opts.voxel_area = train_db.voxel_grid_size
opts.scene_ground = torch.from_numpy(train_db.scene_ground.copy()).float()
opts.min_img_coord = torch.from_numpy(train_db.min_img_coord.copy()).float()
opts.xy_size = torch.from_numpy(train_db.xy_size.copy()).float()



# Model
logger1.info('Building model...')
if args.model == 'gaussian':
    model1 = GaussianVAE(args).to(args.device)
    model2 = GaussianVAE(args).to(args.device)
elif args.model == 'flow':
    model1 = FlowVAE(args).to(args.device)
    model2 = FlowVAE(args).to(args.device)
logger1.info(repr(model1))
logger2.info(repr(model2))
if args.spectral_norm:
    add_spectral_norm(model1, logger=logger)
    add_spectral_norm(model2, logger=logger)

optimizer1 = torch.optim.Adam(model1.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
optimizer2 = torch.optim.Adam(model2.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler1 = get_linear_scheduler(
    optimizer1,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)
scheduler2 = get_linear_scheduler(
    optimizer2,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)



# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    x_tem=batch['template_pc'].to(args.device)
    x_sea=batch['search_pc'].to(args.device)

    # Reset grad and model state
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    #model.train()
    model1.train()
    model2.train()

    if args.spectral_norm:
        spectral_norm_power_iteration(model1, n_power_iterations=1)
        spectral_norm_power_iteration(model2, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss_tem=model1.get_loss(x_tem,kl_weight=kl_weight,writer=writer1,it=it)
    loss_sea=model2.get_loss(x_sea,kl_weight=kl_weight,writer=writer2,it=it)

    # Backward and optimize
    loss_tem.backward()
    loss_sea.backward()
    orig_grad_norm1 = clip_grad_norm_(model1.parameters(), args.max_grad_norm)
    orig_grad_norm2 = clip_grad_norm_(model2.parameters(), args.max_grad_norm)
    optimizer1.step()
    scheduler1.step()
    optimizer2.step()
    scheduler2.step()

    logger1.info('[Train_Tem] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        it, loss_tem.item(), orig_grad_norm1, kl_weight
    ))
    logger2.info('[Train_Sea] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        it, loss_sea.item(), orig_grad_norm2, kl_weight
    ))
    writer1.add_scalar('train_tem/loss', loss_tem, it)
    writer2.add_scalar('train_sea/loss', loss_sea, it)
    writer1.add_scalar('train/kl_weight', kl_weight, it)
    writer2.add_scalar('train/kl_weight', kl_weight, it)
    writer1.add_scalar('train/lr1', optimizer1.param_groups[0]['lr'], it)
    writer2.add_scalar('train/lr2', optimizer2.param_groups[0]['lr'], it)
    writer1.add_scalar('train/grad_norm', orig_grad_norm1, it)
    writer2.add_scalar('train/grad_norm', orig_grad_norm2, it)
    writer1.flush()
    writer2.flush()

def validate_inspect(it):
    z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
    x_tem = model1.sample(z, args.sample_num_points, flexibility=args.flexibility)  # , truncate_std=args.truncate_std)
    x_sea = model2.sample(z, args.sample_num_points, flexibility=args.flexibility)  # , truncate_std=args.truncate_std)
    writer1.add_mesh('val/pointcloud', x_tem, global_step=it)
    writer2.add_mesh('val/pointcloud', x_sea, global_step=it)

    writer1.flush()
    writer2.flush()
    logger1.info('[Inspect] Generating samples...')


def test(it):
    ref_pcs = []
    ref_pcs_tem=[]
    ref_pcs_sea=[]
    for i, data in enumerate(valid_loader):
        if i >= args.test_size:
            break
        ref_pcs_tem.append(data['template_pc'].unsqueeze(0))
        ref_pcs_sea.append(data['search_pc'].unsqueeze(0))
    gen_pcs_tem=[]
    gen_pcs_sea=[]
    for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
        with torch.no_grad():
            z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
            x_tem = model1.sample(z, args.sample_num_points, flexibility=args.flexibility)
            x_sea = model2.sample(z, args.sample_num_points, flexibility=args.flexibility)

            gen_pcs_tem.append(x_tem.detach().cpu())
            gen_pcs_sea.append(x_sea.detach().cpu())
    gen_pcs_tem = torch.cat(gen_pcs_tem, dim=0)[:args.test_size]
    gen_pcs_sea = torch.cat(gen_pcs_sea, dim=0)[:args.test_size]

    with torch.no_grad():
        results_tem = compute_all_metrics(gen_pcs_tem.to(args.device), ref_pcs_tem.to(args.device), args.val_batch_size)
        results_sea = compute_all_metrics(gen_pcs_sea.to(args.device), ref_pcs_sea.to(args.device), args.val_batch_size)
        results_tem = {k: v.item() for k, v in results_tem.items()}
        results_sea = {k: v.item() for k, v in results_sea.items()}
        jsd_tem = jsd_between_point_cloud_sets(gen_pcs_tem.cpu().numpy(), ref_pcs_tem.cpu().numpy())
        jsd_sea = jsd_between_point_cloud_sets(gen_pcs_sea.cpu().numpy(), ref_pcs_sea.cpu().numpy())

        results_tem['jsd'] = jsd_tem
        results_sea['jsd'] = jsd_sea


    writer.add_scalar('test/Coverage_CD', results_tem['lgan_cov-CD'], global_step=it)
    writer.add_scalar('test/MMD_CD', results_tem['lgan_mmd-CD'], global_step=it)
    writer.add_scalar('test/1NN_CD', results_tem['1-NN-CD-acc'], global_step=it)

    writer.add_scalar('test/Coverage_CD', results_sea['lgan_cov-CD'], global_step=it)
    writer.add_scalar('test/MMD_CD', results_sea['lgan_mmd-CD'], global_step=it)
    writer.add_scalar('test/1NN_CD', results_sea['1-NN-CD-acc'], global_step=it)
    writer.add_scalar('test/JSD', results_tem['jsd'], global_step=it)
    writer.add_scalar('test/JSD', results_sea['jsd'], global_step=it)

    logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results_tem['lgan_cov-CD'],))
    logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results_tem['lgan_mmd-CD'],))
    logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results_tem['1-NN-CD-acc'],))
    logger.info('[Test] JsnShnDis | %.6f ' % (results_tem['jsd']))

    logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results_sea['lgan_cov-CD'],))
    logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results_sea['lgan_mmd-CD'],))
    logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results_sea['1-NN-CD-acc'],))
    logger.info('[Test] JsnShnDis | %.6f ' % (results_sea['jsd']))

# Main loop
try:
    it = 1
    while it <= args.max_iters:
    #while it <=200000:
        train(it)
        if it % 2 == 0 or it == args.max_iters:
            validate_inspect(it)

            opt_states1 = {
                'optimizer': optimizer1.state_dict(),
                'scheduler': scheduler1.state_dict(),
            }
            opt_states2 = {
                'optimizer': optimizer2.state_dict(),
                'scheduler': scheduler2.state_dict(),
            }
            ckpt_mgr1.save(model1, args, 0, others=opt_states1, step=it)
            ckpt_mgr2.save(model2, args, 0, others=opt_states2, step=it)

        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
