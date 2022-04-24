import os
import time
import wandb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader
import argparse

from data import ImageDataset, stratified_kfold, get_train_transforms, get_valid_transforms
from model import HappyWhaleModel
from loss import ArcFaceLoss
from train import train_epoch, validate
from utils import seed_everything
from scheduler import CosineAnnealingWarmupRestarts


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(config.seed)
    df = pd.read_csv(config.train_data)
    
    # drop error images
    drop_index = [11604,15881,16782,21966,23306,24862,25895,29468,31831,35805,37176,40834,47480,48455,36710,47161,23626]
    df = df.drop(drop_index, axis=0).reset_index(drop=True)
    df.species.replace({"globis": "short_finned_pilot_whale",
                      "pilot_whale": "short_finned_pilot_whale",
                      "kiler_whale": "killer_whale",
                      "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)

    classes = df.individual_id.unique()
    inv_class_map = dict(enumerate(classes, 0))
    class_map = {v: k for k, v in inv_class_map.items()}
    df.individual_id = df.individual_id.map(class_map)
    
    # divide df by id counts
    df1 = df[df['individual_id'].map(df['individual_id'].value_counts()) == 1]
    df2 = df[df['individual_id'].map(df['individual_id'].value_counts()) > 1]

    train_one, valid_one = stratified_kfold(df=df1, fold=config.fold, n_split=config.n_split, seed=config.seed, target_col='species')
    train_two, valid_two = stratified_kfold(df=df2, fold=config.fold, n_split=config.n_split, seed=config.seed, target_col='individual_id')

    train_one = np.take(df1.index.to_numpy(), train_one)    
    train_two = np.take(df2.index.to_numpy(), train_two)
    valid_two = np.take(df2.index.to_numpy(), valid_two)

    train_index = np.sort(np.concatenate([train_one, train_two], axis=0))
    valid_index = valid_two # valid_one has no validation ids

    assert len(train_one) + len(train_two) == len(set(train_index))

    x_data, y_data = df['image'].values, df['individual_id'].values
    x_train, y_train = x_data[train_index], y_data[train_index]
    x_valid, y_valid = x_data[valid_index], y_data[valid_index]

    train_transform = get_train_transforms(config.image_size)
    valid_transform = get_valid_transforms(config.image_size)

    train_set = ImageDataset(path=x_train, target=y_train, transform=train_transform, root=config.train_image_root)
    valid_set = ImageDataset(path=x_valid, target=y_valid, transform=valid_transform, root=config.train_image_root)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = HappyWhaleModel(model_name=config.model_name, emb_size=config.emb_size).to(device)

    loss_fn = ArcFaceLoss(s=config.s, m=config.m)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler()

    cosine_annealing_scheduler_arg = dict(
        first_cycle_steps=len(train_set)//config.batch_size*config.epoch,
        cycle_mult=1.0,
        max_lr=config.learning_rate,
        min_lr=1e-06,
        warmup_steps=len(train_set)//config.batch_size*3, # warm up 0~3 epoch
        gamma=0.9
    )
    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cosine_annealing_scheduler_arg)

    best_acc = 0
    best_model = None

    for i in range(config.epoch):
        print(f"epoch: {i}")

        train_loss, train_acc = train_epoch(
            model, optimizer, loss_fn, train_loader, scheduler, grad_scaler, config.iters_to_accumulate)

        valid_loss, valid_acc = validate(model, loss_fn, valid_loader)

        print(f"train loss {train_loss :.4f} acc {train_acc :.4f}")
        print(f"valid loss {valid_loss :.4f} acc {valid_acc :.4f}")

        if best_acc < valid_acc:
            best_acc = valid_acc
            best_model = deepcopy(model.state_dict())

    return best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Config')
    
    parser.add_argment('--n_split', type=int, default=5)
    parser.add_argment('--fold', type=int, default=0)
    parser.add_argment('--seed', type=int, default=2022)
    parser.add_argment('--epoch', type=int, default=21)
    parser.add_argment('--model_name', type=str, default='tf_efficientnet_b5_ns')
    parser.add_argment('--train_data', type=str, default='../data/train.csv')
    parser.add_argment('--train_image_root', type=str, default='../data/train_detec_512_v5/')
    parser.add_argment('--image_size', type=tuple, default=(16, 16))
    parser.add_argment('--batch_size', type=int, default=512)
    parser.add_argment('--iters_to_accumulate', type=int, default=1)
    parser.add_argment('--learning_rate', type=float, default=6.25e-6)
    parser.add_argment('--emb_size', type=int, default=2048)
    parser.add_argment('--margin', type=float, default=0.55)
    parser.add_argment('--s', type=float, default=30.0)
    parser.add_argment('--m', type=float, default=0.5)
    parser.add_argment('--weight_decay', type=tuple, default=0.)
    parser.add_argment('--experiment', type=str, default='test')
    
    config = parser.parse_args()
    
    config.learning_rate = config.learning_rate * config.batch_size * config.iters_to_accumulate

    main(config)
    