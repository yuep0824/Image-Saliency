import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FCN8s_Baseline, UNet_Baseline
from utils import SaliencyDataset, eval_metrics, save_prediction, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Saliency Prediction Baseline')
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--model', type=str, default='unet', choices=['fcn', 'unet'], help='选择模型')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='output', help='输出保存目录')
    parser.add_argument('--save_freq', type=int, default=10, help='模型保存频率')
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}')
    
    # 适配返回值：img, mask, (ori_h, ori_w), mask_ori, img_ori
    for imgs, masks, _, _, _ in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计损失
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    print(f'Train Epoch {epoch} | Avg Loss: {avg_loss:.4f}')
    return avg_loss

# ---------------------- 验证函数（适配多返回值+保存原始尺寸预测） ----------------------
def validate(model, loader, criterion, device, save_dir):
    model.eval()
    total_loss = 0.0
    preds = []
    gts = []
    pbar = tqdm(loader, desc='Validate')
    
    with torch.no_grad():
        for imgs, masks, ori_sizes, mask_oris, img_oris in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # 转numpy计算指标（用resize后的掩码）
            outputs_np = outputs.squeeze(1).cpu().numpy()  # [B, H, W]
            masks_np = masks.squeeze(1).cpu().numpy()      # [B, H, W]
            preds.extend(outputs_np)
            gts.extend(masks_np)
            
            # 保存预测结果（恢复到原始尺寸）
            for i, (ori_size, img_ori) in enumerate(zip(ori_sizes, img_oris)):
                # 生成保存文件名（基于原始图像路径）
                img_name = f"pred_{i}_{epoch}.png" if 'epoch' in locals() else f"pred_{i}.png"
                save_path = os.path.join(save_dir, 'preds', img_name)
                save_prediction(outputs_np[i], save_path, ori_size)
    
    # 计算指标
    avg_loss = total_loss / len(loader)
    avg_cc, avg_kl = eval_metrics(preds, gts)
    print(f'Validate | Avg Loss: {avg_loss:.4f} | CC: {avg_cc:.4f} | KL Div: {avg_kl:.4f}')
    return avg_loss, avg_cc, avg_kl


def main():
    args = parse_args()
    device = get_device()
    print(f'Using device: {device}')
    
    # 创建保存目录
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
    elif args.model == 'unet':
        model = UNet_Baseline(n_channels=3, n_classes=1).to(device)
    print(f'Loaded {args.model.upper()} model')
    
    # 3. 损失函数与优化器（回归任务用MSE）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. 训练+验证循环
    best_cc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_cc, val_kl = validate(model, val_loader, criterion, device, args.save_dir)
        
        # 保存最优模型（按CC指标）
        if val_cc > best_cc:
            best_cc = val_cc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'models', 'best_model.pth'))
            print(f'Best model saved (CC: {best_cc:.4f})')
        
        # 按频率保存模型
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'models', f'epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()