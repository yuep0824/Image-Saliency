
# ---------------------- 类激活热图分析（Grad-CAM） ----------------------
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import cv2
import random
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 获取目标层
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target=None):

        output = self.model(input_tensor)
        if target is None:
            target = output.mean()  # 显著性预测任务中，使用输出的平均值
        
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # 计算权重（全局平均池化）
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和
        cam = torch.sum(weights * activations, dim=1)  # [1, H, W]
        cam = torch.relu(cam)  # ReLU激活（只关注正贡献）
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), output.detach()
        
    def visualize(self, input_tensor, original_tensor=None, save_path=None, img_size=(256, 256)):
        
        cam, _ = self.generate_cam(input_tensor)
        
        # 如果有原始图像，叠加显示
        if original_tensor is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # tensor形状: [1, 3, H, W] -> [H, W, 3]
            img_to_show = original_tensor[0].cpu().numpy().transpose(1, 2, 0)
            # 反归一化：从[0,1]到[0,255]
            img_to_show = (img_to_show * 255).astype(np.uint8)
            
            # 将热图缩放到模型输入尺寸
            cam_resized = cv2.resize(cam, (img_size[1], img_size[0]))
            
            # 原始图像（已经过与模型输入相同的预处理）
            axes[0].imshow(img_to_show)
            axes[0].set_title('Model Input Image')
            axes[0].axis('off')
            
            # CAM热图（缩放到模型输入尺寸）
            axes[1].imshow(cam_resized, cmap='hot')
            axes[1].set_title('CAM Heatmap')
            axes[1].axis('off')
            
            # 叠加显示
            axes[2].imshow(img_to_show)
            axes[2].imshow(cam_resized, cmap='hot', alpha=0.4)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            
        else:
            # 只显示热图（调整到模型输入尺寸）
            cam_resized = cv2.resize(cam, (img_size[1], img_size[0]))
            plt.figure(figsize=(10, 8))
            plt.imshow(cam_resized, cmap='jet')
            plt.colorbar(label='Activation Strength')
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved CAM visualization to {save_path}")
        
        #plt.show()
        plt.close()
        return cam


def get_target_layer_name(model_type, model):
    if model_type == 'fcn':
        # FCN模型的目标层：最后一个卷积层
        return 'features3'
    elif model_type == 'unet':
        # UNet模型的目标层：解码器中的最后一个卷积块
        return 'up4.conv.double_conv.4'  # 最后一个ReLU层
    elif model_type == 'fcn_enhance':
        # EnhancedFCN：使用encoder3第二个卷积层
        return 'fuse_conv' 
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_cam_visualizations(model, dataloader, device, save_dir, model_type='unet', num_samples=16, img_size=(256, 256)):

    model.eval()
    
    cam_dir = os.path.join(save_dir, 'cam_visualizations')
    os.makedirs(cam_dir, exist_ok=True)

    random.seed(42)
    
    # 获取目标层名称
    target_layer = get_target_layer_name(model_type, model)
    
    # 初始化Grad-CAM
    cam_generator = GradCAM(model, target_layer)
    
    # 获取样本
    data_iter = iter(dataloader)
    
    for i in range(min(num_samples, len(dataloader))):
        try:
            # 获取一个批次
            imgs, _ , _, _ , _ = next(data_iter)
            
            '''只处理第一个样本
            img_tensor = imgs[0:1].to(device)
            original_img = img_oris[0]
            '''
            # 随机选择一个样本索引
            batch_size = imgs.size(0)
            random_idx = random.randint(0, batch_size - 1)
            img_tensor = imgs[random_idx:random_idx+1].to(device)
            original_tensor = imgs[random_idx:random_idx+1]
            
            # 生成可视化
            save_path = os.path.join(cam_dir, model_type+f'_cam_sample_{i+1}.png')
            cam_generator.visualize(
                img_tensor, 
                original_tensor=original_tensor,  # 使用预处理后的tensor
                save_path=save_path, 
                img_size=img_size
            )
            
            print(f"Generated CAM visualization for sample {i+1}")
            
        except StopIteration:
            break
    
    print(f"All CAM visualizations saved to {cam_dir}")