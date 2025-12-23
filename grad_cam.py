
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

        # 如果梯度是3维，重塑为4维
        if gradients.dim() == 3:
            B, L, C = gradients.shape
            H = W = int(L ** 0.5)
            gradients = gradients.permute(0, 2, 1).reshape(B, C, H, W)
            activations = activations.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 计算权重（全局平均池化）
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和
        cam = torch.sum(weights * activations, dim=1)  # [1, H, W]
        cam = torch.relu(cam)  # ReLU激活（只关注正贡献）
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), output.detach()
        
    def visualize(self, model_type, input_tensor, original_tensor=None, save_path=None, img_size=(256, 256)):
        
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

            fig = plt.figure(figsize=(12, 5))
            
            # 定义参数
            image_width = 0.30  # 每个图像的宽度
            cbar_width = 0.015  # colorbar宽度
            gap = 0.01  # 元素之间的间距
            
            # 计算每个元素的水平起始位置
            x0_img1 = 0.07
            x0_img2 = x0_img1 + image_width + gap
            x0_img3 = x0_img2 + image_width + gap
            x0_cbar3 = x0_img3 + image_width + 0.01
            
            # 垂直位置 - 只考虑图像显示区域，不包括标题
            y0_images = 0.12  # 图像底部位置
            image_height = 0.72  # 图像高度
            
            # 创建图像轴
            ax1 = fig.add_axes([x0_img1, y0_images, image_width, image_height])
            ax2 = fig.add_axes([x0_img2, y0_images, image_width, image_height])
            ax3 = fig.add_axes([x0_img3, y0_images, image_width, image_height])
            
            
            # 原始图像（已经过与模型输入相同的预处理）
            ax1.imshow(img_to_show)
            ax1.set_title('Model Input Image',fontsize=12)
            ax1.axis('off')
            
            # CAM热图（缩放到模型输入尺寸）
            im_cam = ax2.imshow(cam_resized, cmap='hot')
            ax2.set_title(f'CAM Heatmap ({model_type})',fontsize=12)
            ax2.axis('off')
            
            # 叠加显示
            ax3.imshow(img_to_show)
            ax3.imshow(cam_resized, cmap='hot', alpha=0.4)
            ax3.set_title('Overlay',fontsize=12)
            ax3.axis('off')

            # 添加colorbar
            cax3 = fig.add_axes([x0_cbar3, ax3.get_position().y0, cbar_width, ax3.get_position().height])
            fig.colorbar(im_cam, cax=cax3)
            cax3.axis('off')
            
            
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
            plt.close()
            print(f"Saved CAM visualization to {save_path}")
            
        
        #plt.show()
        
        return cam


def get_target_layer_name(model_type, model):
    if model_type == 'fcn':
        # FCN模型的目标层：最后一个卷积层
        return 'upsample8'
    elif model_type == 'fcn_enhance':
        # EnhancedFCN：使用融合层
        return 'fuse_conv'
    elif model_type == 'unet':
        # UNet模型的目标层：解码器中的最后一个卷积块
        return 'up4.conv.double_conv.4'  # 最后一个ReLU层
    elif model_type == 'unet-resnet18':
        return 'decoder2'
    elif model_type == 'mobilenetv3-unet':
        return 'up1.1'
    elif model_type == 'swin-unet':
        return 'up_block2'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_cam_visualizations(model, dataloader, device, save_dir, model_type='unet', num_samples=16, img_size=(256, 256)):

    model.eval()
    
    cam_dir = os.path.join(save_dir, 'cam_visualizations', model_type)
    os.makedirs(cam_dir, exist_ok=True)

    random.seed(40)
    
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
                model_type,
                img_tensor, 
                original_tensor=original_tensor,  # 使用预处理后的tensor
                save_path=save_path, 
                img_size=img_size
            )
            
            print(f"Generated CAM visualization for sample {i+1}")
            
        except StopIteration:
            break
    
    print(f"All CAM visualizations saved to {cam_dir}")


