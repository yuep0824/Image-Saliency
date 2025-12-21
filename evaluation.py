
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import SaliencyDataset, get_device
from model import FCN8s_Baseline, UNet_Baseline, EnhancedFCN
from grad_cam import save_cam_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--model', type=str, default='unet', choices=['fcn', 'unet', 'fcn_enhance'], help='选择模型')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='output', help='输出保存目录')
    parser.add_argument('--save_freq', type=int, default=10, help='模型保存频率')
    return parser.parse_args()

def load_best_model(args, device):
    print(f"\n{'='*60}")
    print(f"Best model {args.model.upper()} loading...")
    
    # 根据模型类型创建模型架构
    if args.model == 'fcn':
        model = FCN8s_Baseline(num_classes=1)
    elif args.model == 'unet':
        model = UNet_Baseline(n_channels=3, n_classes=1)
    elif args.model == 'fcn_enhance':
        model = EnhancedFCN(num_classes=1)
    print(f'Loaded {args.model.upper()} model')
    
    # 加载模型权重
    model_path = os.path.join(args.save_dir, 'models', args.model+'_best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
        
    return model
    
def load_valid_data(args, device):
    print(f"\n{'='*60}")
    print("Validation dataset loading...")
    
    val_dataset = SaliencyDataset(
        root_dir=os.path.join(args.data_root, '3-Saliency-TestSet'),
        img_size=args.img_size,
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return val_loader, val_dataset

def main():
    args = parse_args()
    
    '''Initialization'''
    args.data_root = '/root/autodl-tmp/Pro3/data'
    args.model = 'fcn'
    args.batch_size = 16
    args.save_dir = '/root/autodl-tmp/Pro3/'
    
    device = get_device()
    print(f'Using device: {device}')
    
    val_loader, _ = load_valid_data(args, device)
    model = load_best_model(args, device)
    save_cam_visualizations(
        model, 
        val_loader, 
        device, 
        args.save_dir, 
        args.model,
        10,
        args.img_size
    )
    
if __name__ == '__main__':
    main()