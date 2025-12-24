
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import SaliencyDataset, get_device, visualize_predictions, plot_all_training_metrics, analyze_by_category_simple
from model import FCN8s_Baseline, EnhancedFCN, UNet_Baseline, UNet_ResNet18, MobileNetV3_UNet, SwinUNet
from grad_cam import save_cam_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--data_root', type=str, default='data', help='数据集根目录')
    parser.add_argument('--model', type=str, default='mobilenetv3-unet', choices=['fcn', 'fcn_enhance', 'unet', 'unet-resnet18', 'mobilenetv3-unet', 'swin-unet'], help='选择模型')
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

def categories_analysis(args, device):
    print("\n" + "="*60)
    print("加载所有六个模型...")
    print("="*60)
    
    model_dict = {}
    model_types = ['fcn', 'fcn_enhance', 'unet', 'unet-resnet18', 'mobilenetv3-unet', 'swin-unet']
    
    for model_type in model_types:
        args.model = model_type
        model_dict[model_type] = load_best_model(args, device)
        
    print("\n" + "="*60)
    print("开始按图像类别分析模型性能...")
    print("="*60)
    
    results = analyze_by_category_simple(
        model_dict=model_dict,
        data_root=args.data_root,
        img_size=args.img_size,
        device=device,
        save_dir=os.path.join(args.save_dir, 'category_analysis')
    )
    return

    
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
    '''
    save_cam_visualizations(
        model, 
        val_loader, 
        device, 
        args.save_dir, 
        args.model,
        len(val_loader),
        args.img_size
    )
    
    visualize_predictions(
        model,
        val_loader,
        device,
        args.save_dir,
        args.model,
        len(val_loader)  
    )
    
    # 设置日志目录
    log_dir = os.path.join(args.save_dir, 'logs')
    
    if os.path.exists(log_dir):
        print(f"\nLogfiles analyzing: {log_dir}")
        
        # 设置输出目录
        output_dir = os.path.join(args.save_dir, 'training_analysis')
        
        # 绘制所有指标
        plot_all_training_metrics(log_dir, output_dir)
    else:
        print(f"Warning: The direction of logfiles is not exist:{log_dir}")
    '''
    
    categories_analysis(args, device)
    
if __name__ == '__main__':
    main()