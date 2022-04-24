import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, optimizer, loss_fn, loader, scheduler, scaler, iters_to_accumulate=1):
    model.train()
    
    losses, y_true, y_pred, embed, species = [], [], [], [], []
    for i, (x, y) in enumerate(tqdm(loader)):
        x, y = x.to(device), y.to(device)
        
        with torch.cuda.amp.autocast():
            output, _ = model(x, y) 
            loss = loss_fn(output, y)
        scaler.scale(loss).backward()

        if ((i + 1) % iters_to_accumulate == 0) or ((i + 1) == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()

        y_true.extend(y.detach().cpu())
        y_pred.extend(output.detach().cpu().argmax(-1))
        losses.append(loss.detach().cpu().item())

    y_true_stack = np.stack(y_true, axis=0)
    y_pred_stack = np.stack(y_pred, axis=0)
    acc = accuracy_score(y_true_stack, y_pred_stack)
    
    return np.mean(losses), acc

def validate(model, loss_fn, loader):
    model.eval()

    losses, y_true, y_pred = [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)

            output, _ = model(x, y) 

            loss = loss_fn(output, y)

            y_true.extend(y.detach().cpu())
            y_pred.extend(output.detach().cpu().argmax(-1))      

    y_true_stack = np.stack(y_true, axis=0)
    y_pred_stack = np.stack(y_pred, axis=0)

    acc = accuracy_score(y_true_stack, y_pred_stack)

    return np.mean(losses), acc