import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.hopf_model import HopfCNN
from utils.dataset import MineDataset
from utils.transforms import get_train_transforms, get_val_transforms
from utils.metrics import AverageMeter, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Hopf-CNN Training')
    parser.add_argument('--data_root', default='./data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='mps', choices=['cuda', 'cpu'])
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--resume', default=None, help='Checkpoint to resume from')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency')
    return parser.parse_args()

def setup_environment(args):
    torch.manual_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    return device

def create_model(device):
    model = HopfCNN(
        in_channels=3,
        num_classes=2,
        base_channels=64,
        hopf_params={
            'dt': 0.025,
            'T': 8,
            'alpha': 0.9,
            'init_freq': 1.0
        }
    )
    return model.to(device)

def create_datasets(data_root):
    train_ds = MineDataset(
        root_dir=os.path.join(data_root, 'train'),
        transform=get_train_transforms()
    )
    val_ds = MineDataset(
        root_dir=os.path.join(data_root, 'val'),
        transform=get_val_transforms()
    )
    return train_ds, val_ds

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    time_meter = AverageMeter()

    end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs, dynamics = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        prec1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        time_meter.update(time.time() - end)
        end = time.time()

        # Log dynamics
        if i % 50 == 0 and writer is not None:
            writer.add_scalar('train/batch_loss', loss.item(), epoch * len(loader) + i)
            writer.add_scalar('train/batch_acc', prec1.item(), epoch * len(loader) + i)
            if dynamics:
                writer.add_histogram('dynamics/amplitude', dynamics['amplitude'], epoch)
                writer.add_histogram('dynamics/phase', dynamics['phase'], epoch)

    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc', top1.avg, epoch)
    
    print(f'Epoch: {epoch} | Train Loss: {losses.avg:.4f} | Acc: {top1.avg:.2f}% | Time: {time_meter.sum:.2f}s')

def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/acc', top1.avg, epoch)
    
    print(f'Validation | Loss: {losses.avg:.4f} | Acc: {top1.avg:.2f}%')
    return losses.avg, top1.avg

def save_checkpoint(state, is_best, save_dir):
    filename = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_file)

def main():
    args = parse_args()
    device = setup_environment(args)
    
    # Initialize components
    model = create_model(device)
    train_ds, val_ds = create_datasets(args.data_root)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    writer = SummaryWriter(log_dir=args.log_dir)
    best_acc = 0
    
    # Resume if needed
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Training loop
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Validation
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
            
            # Save checkpoint
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.save_dir)
        
        scheduler.step()
    
    writer.close()
    print(f'Training completed. Best validation accuracy: {best_acc:.2f}%')

if name == 'main':
    main()
