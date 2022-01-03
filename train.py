"""
Implementation of Flow Contrastive Estimation (FCE) on 2D dataset.
https://arxiv.org/abs/1912.00589
"""
import os
import math
import argparse
import torch
from ebm import EBM
import util

import wandb


parser = argparse.ArgumentParser(description='Flow Contrastive Estimation of Energy Based Model')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--epoch', default=100, type=int, help='number of training epochs')
parser.add_argument('--flow', default='glow', type=str, help='Flow model to use')
parser.add_argument('--threshold', default=0.6, type=float, help='threshold for alternate training')
parser.add_argument('--batch', default=1000, type=int, help='batch size')
parser.add_argument('--dataset', default='8gaussians', type=str, choices=['8gaussians', 'spiral', '2spirals', 'checkerboard', 'rings', 'pinwheel'], help='2D dataset to use') 
parser.add_argument('--samples', default=500000, type=int, help='number of 2D samples for training')
parser.add_argument('--lr_ebm', default=1e-3, type=float, help='learning rate for EBM')
parser.add_argument('--lr_flow', default=7e-4, type=float, help='learning rate for Flow')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
args = parser.parse_args()

wandb.init(project='FCE-2d')
wandb.config.update(args)

device = torch.device('cuda')
torch.manual_seed(args.seed)
# ------------------------------
# I. MODELS
# ------------------------------
energy = EBM().to(device)
if args.flow == 'glow':
    from flows.glow import Glow
    flow = Glow(width=64, depth=5, n_levels=1, data_dim=2).to(device)
elif args.flow == 'maf':
    from flows.maf import MAF
    flow = MAF(n_blocks=5, input_size=2, hidden_size=100, n_hidden=1).to(device)
# ------------------------------
# II. OPTIMIZERS
# ------------------------------
optim_energy = torch.optim.Adam(energy.parameters(), lr=args.lr_ebm, betas=(args.b1, args.b2))
optim_flow = torch.optim.Adam(flow.parameters(), lr=args.lr_flow, betas=(args.b1, args.b2))
# ------------------------------
# III. DATA LOADER
# ------------------------------
dataset, dataloader = util.get_data(args)
dataset = dataset.to(device)
# ------------------------------
# IV. TRAINING
# ------------------------------
wandb.watch(energy)
wandb.watch(flow)

def main(args):
    train_energy = True
    for epoch in range(args.epoch):
        for i, x in enumerate(dataloader):           
            x = x.to(device)
            # -----------------------------
            #  Generate noise
            # -----------------------------
            z = flow.base_dist.sample((args.batch,))
            # -----------------------------
            #  Train Energy Model
            # -----------------------------
            if train_energy:
                optim_energy.zero_grad()
                loss_energy, acc  = util.value(energy, flow, x, z, maximize=True)
                loss_energy.backward()
                optim_energy.step()  
            # -----------------------------
            #  Train Flow Model
            # -----------------------------
            else:
                optim_flow.zero_grad() 
                loss_flow, acc = util.value(energy, flow, x, z, maximize=False)
                loss_flow.backward()
                optim_flow.step()

            wandb.log({'epoch': epoch,
                       'value': loss_energy.item() if train_energy else -loss_flow.item(),
                       'acc': acc,
                       'mse': util.mse(energy, util.MixedGaussian(device=device), dataset),  # comment out if not using 8gaussians
                })


            if acc > args.threshold:
                train_energy = False
            else:
                train_energy = True


        # Save checkpoint
        print('Saving models...')
        state = {
        'energy': energy.state_dict(),
        'flow': flow.state_dict(),
        'value': loss_energy,
        'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        ckpts = 'ckpts/fce-{}-2d-{}.pth.tar'.format(args.flow, args.dataset)
        torch.save(state, ckpts)

        # visualization
        # util.plot(dataset, energy, flow, epoch, device)
        




if __name__ == '__main__':
    print(args)
    main(args)
