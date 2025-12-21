import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from model import FCN8s_Baseline, EnhancedFCN, UNet_Baseline, UNet_ResNet18, MobileNetV3_UNet, SwinUNet
from utils import SaliencyDataset, eval_metrics, save_prediction, get_device
from grad_cam import save_cam_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description='Saliency Prediction Baseline')
    parser.add_argument('--data_root', type=str, default='./data', help='训练集根目录')
    parser.add_argument('--model', type=str, default='mobilenetv3-unet', choices=['fcn', 'fcn_enhance', 'unet', 'unet-resnet18', 'mobilenetv3-unet', 'swin-unet'], help='选择模型')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='output', help='输出保存目录')
    parser.add_argument('--save_freq', type=int, default=10, help='模型保存频率')
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}')
    
    for imgs, masks, _, _, _ in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    print(f'Train Epoch {epoch} | Avg Loss: {avg_loss:.4f}')
    return avg_loss


def validate(model, loader, criterion, device, save_dir, epoch, model_name):
    model.eval()
    total_loss = 0.0
    preds = []
    gts = []
    pbar = tqdm(loader, desc='Validate')
    
    with torch.no_grad():
        for imgs, masks, ori_sizes, mask_oris, img_oris in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            outputs_np = outputs.squeeze(1).cpu().numpy()  # [B, H, W]
            masks_np = masks.squeeze(1).cpu().numpy()      # [B, H, W]
            preds.extend(outputs_np)
            gts.extend(masks_np)
            
            # 保存预测结果（恢复到原始尺寸）
            for i, (ori_size, img_ori) in enumerate(zip(ori_sizes, img_oris)):
                target_dir = os.path.join(save_dir, f'preds/{model_name}')
                os.makedirs(target_dir, exist_ok=True)
                
                img_name = f"pred_{i}_{epoch}.png" if 'epoch' in locals() else f"pred_{i}.png"
                save_path = os.path.join(target_dir, img_name)
                save_prediction(outputs_np[i], save_path, ori_size)
                
    avg_loss = total_loss / len(loader)
    avg_cc, avg_kl = eval_metrics(preds, gts)
    print(f'Validate | Avg Loss: {avg_loss:.4f} | CC: {avg_cc:.4f} | KL Div: {avg_kl:.4f}')
    return avg_loss, avg_cc, avg_kl


def main():
    args = parse_args()
    device = get_device()
    print(f'Using device: {device}')
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'preds'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)
    
    # 1. 加载数据
    train_dataset = SaliencyDataset(
        root_dir=os.path.join(args.data_root, '3-Saliency-TrainSet'),
        img_size=args.img_size
    )
    val_dataset = SaliencyDataset(
        root_dir=os.path.join(args.data_root, '3-Saliency-TestSet'),
        img_size=args.img_size,
        is_train=False
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 初始化模型
    if args.model == 'fcn':
        model = FCN8s_Baseline(num_classes=1).to(device)
    elif args.model == 'fcn_enhance':
        model = EnhancedFCN(num_classes=1).to(device)
    elif args.model == 'unet':
        model = UNet_Baseline(n_channels=3, n_classes=1).to(device)
    elif args.model == 'unet-resnet18':
        model = UNet_ResNet18().to(device)
    elif args.model == 'mobilenetv3-unet':
        model = MobileNetV3_UNet(n_channels=3, n_classes=1).to(device)
    elif args.model == 'swin-unet':
        model = SwinUNet(img_size=args.img_size[0], in_chans=3, num_classes=1).to(device)
    print(f'Loaded {args.model.upper()} model')
    
    # 3. 损失函数与优化器（回归任务用MSE）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',           # 监控CC指标（越大越好）
        factor=0.7,           # 学习率乘以0.5
        patience=3,           # 3个epoch没有改善就降低学习率
        verbose=True,         # 打印调整信息
        threshold=0.001,      # 改善至少0.001才算有效
        threshold_mode='abs', # 绝对改善阈值
        min_lr=1e-6           # 最小学习率
    )

    # 4. 训练+验证循环
    log_dir = f'./logs/{args.model}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    best_cc = 0.0
    early_stop_counter = 0
    early_stop_patience = 10  # 10个epoch没有改善就早停
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_cc, val_kl = validate(model, val_loader, criterion, device, args.save_dir, epoch, args.model)
        
        # 更新学习率调度器（基于CC指标）
        scheduler.step(val_cc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch} | current lr: {current_lr:.2e}')

        # tensorboard 可视化
        writer.add_scalar('Loss/train', train_loss, epoch)      # 训练损失
        writer.add_scalar('Loss/validation', val_loss, epoch)   # 验证损失
        writer.add_scalar('Metric/CC', val_cc, epoch)           # CC值
        writer.add_scalar('Metric/KL Divergence', val_kl, epoch)      # KL散度  
        
        # 保存最优模型（按CC指标）
        if val_cc > best_cc:
            best_cc = val_cc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'models', args.model+'_best_model.pth'))
            print(f'Best model saved (CC: {best_cc:.4f})')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 按频率保存模型
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'models', args.model+f'_epoch_{epoch}.pth'))
            
        # 早停机制
        if early_stop_counter >= early_stop_patience:
            print(f' CC did not improve within {early_stop_patience} epochs.')
            break
            
    writer.close()

    # 5. Grad_cam生成保存
    save_cam_visualizations(
        model, 
        val_loader, 
        device, 
        args.save_dir, 
        args.model,
        img_size=args.img_size
    )

if __name__ == '__main__':
    main()