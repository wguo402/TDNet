import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)#(128,2048,3)---->(128,256)  batch_size & feature
        return code#([128, 256])

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x):
        code = self.encode(x)#([128, 2048, 3])----->(128,256)   FC only, just for feature extraction
        loss = self.diffusion.get_loss(x, code)#([128, 2048, 3])&(128,256)--->    A raw and a processed loss
        return loss