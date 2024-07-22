import argparse
import json
import os
from tqdm import tqdm
import multiprocessing
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from backbone import ResNetBackbone, MobileNetBackbone
from mtp import multiple_trajectory_model, multiple_trajectory_loss
import util
from datasets import NuScenesDataset  # Import the dataset class

def main():
    # Argument parsing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str, help='experiment name. saved to /exps/[name]')
    parser.add_argument('--max_epoc', default=50, type=int)
    parser.add_argument('--min_loss', default=0.56234, type=float, help='minimum loss threshold that training step')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=8, type=int)  # Set num_workers to 0 to avoid multiprocessing issues
    parser.add_argument('--optimizer', default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--gpu_idx', default='0,1', type=str, help=' ids of gpus')
    parser.add_argument('--tsboard', action='store_true', help='To visualize experiments')

    parser.add_argument('--num_modes', default=2)
    parser.add_argument('--backbone', default='mobilenet_v2', choices=['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--unfreeze', default=0, type=int, help='number of backbone layers to update weight')

    args = parser.parse_args()

    exp_path, train_path, val_path, infer_path, ckpt_path = util.make_path(args)

    f = open(ckpt_path + '/' + 'exp_config.txt', 'w')
    json.dump(args.__dict__, f, indent=2)
    f.close()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainset = NuScenesDataset('./dataset', 'mini_train')
    valset = NuScenesDataset('./dataset', 'mini_val')
    
    train_loader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=0, multiprocessing_context='spawn')
    val_loader = DataLoader(valset, shuffle=True, batch_size=args.batch_size, num_workers=0, multiprocessing_context='spawn')

    # train_loader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    # val_loader = DataLoader(valset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    backbone = ResNetBackbone(args.backbone) if args.backbone.startswith('resnet') else MobileNetBackbone(args.backbone)

    total_layer_ct = sum(1 for _ in backbone.parameters())
    for i, param in enumerate(backbone.parameters()):
        if i < total_layer_ct - args.unfreeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model = multiple_trajectory_model(backbone, args.num_modes)
    loss_function = multiple_trajectory_loss(args.num_modes, 1, 6)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else optim.SGD(model.parameters(), lr=args.lr)

    torch.save(model, ckpt_path + '/' + 'model.archi')
    torch.save(optimizer, ckpt_path + '/' + 'optim.archi')

    model = nn.DataParallel(model)
    model = model.to(device)

    current_ep_loss = 10000

    for epoch in range(args.max_epoc):
        print("Commencing Training")
        model.train()
        loss_tr_mean = []
        for img, state, gt in tqdm(train_loader):
            img, state, gt = util.NaN2Zero(img), util.NaN2Zero(state), util.NaN2Zero(gt)
            img, state, gt = img.to(device), state.to(device), gt.to(device)

            optimizer.zero_grad()
            prediction = model(img, state)
            loss = loss_function(prediction, gt.unsqueeze(1))

            loss.backward()
            optimizer.step()
            loss_tr_mean.append(loss.item())

        print("Commencing Validation")
        model.eval()
        loss_val_mean = []
        for img, state, gt in tqdm(val_loader):
            img, state, gt = util.NaN2Zero(img), util.NaN2Zero(state), util.NaN2Zero(gt)
            img, state, gt = img.to(device), state.to(device), gt.to(device)

            prediction = model(img, state)
            loss = loss_function(prediction, gt.unsqueeze(1))

            loss_val_mean.append(loss.item())

        ep_loss_tr, ep_loss_val = np.mean(loss_tr_mean), np.mean(loss_val_mean)
        print(f"Current Training Loss is: {ep_loss_tr:.4f}")
        print(f"Current Validation Loss is: {ep_loss_val:.4f}")

        checkpoint = {'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': ep_loss_tr, 'ep': epoch}

        if ep_loss_val < current_ep_loss:
            print("Best Validation Loss Achieved")
            torch.save(checkpoint, ckpt_path + '/' + 'weight_best.pth')
            current_ep_loss = ep_loss_val
            with open(ckpt_path + '/' + 'save-log.txt', 'a') as f:
                f.write(f'\n loss {ep_loss_val:.3f} achieved at epoch {epoch:d}')

        if np.allclose(ep_loss_val, args.min_loss, atol=1e-4):
            print(f"Achieved Loss under min_loss after {epoch} iterations.")
            torch.save(checkpoint, ckpt_path + '/' + f'weight_{ep_loss_val:.3f}.pth')
            break

    torch.save(checkpoint, ckpt_path + '/' + 'weight_last.pth')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
