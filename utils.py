import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import stats
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------- 替换为你的SaliencyDataset（修复潜在问题） ----------------------
class SaliencyDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        # 递归获取所有图像路径（Stimuli目录下）
        self.img_paths = []
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
        for root, _, files in os.walk(os.path.join(root_dir, "Stimuli")):
            for file in files:
                if file.lower().endswith(img_extensions):
                    self.img_paths.append(os.path.join(root, file))

        # 匹配掩码路径（替换为FIXATIONMAPS，自动匹配扩展名）
        self.mask_paths = []
        for img_path in self.img_paths:
            mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")
            mask_path_base = os.path.splitext(mask_path)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = mask_path_base + ext
                if os.path.exists(candidate):
                    self.mask_paths.append(candidate)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"未找到{img_path}对应的掩码文件")

        # 修复：数据增强应作用于PIL Image（先转PIL再增强，避免tensor输入报错）
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ]) if is_train else None

        print(f"成功加载{len(self.img_paths)}个样本（{root_dir}）")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 加载原始图像
        img_path = self.img_paths[idx]
        img_ori = cv2.imread(img_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)  # BGR→RGB
        ori_h, ori_w = img_ori.shape[:2]

        # 2. 加载原始掩码（单通道）
        mask_path = self.mask_paths[idx]
        mask_ori = cv2.imread(mask_path, 0)  # 0表示单通道灰度图

        # 3. 调整尺寸
        img = cv2.resize(img_ori, self.img_size)
        mask = cv2.resize(mask_ori, self.img_size)

        # 4. 转为PIL Image（用于数据增强）
        img_pil = transforms.ToPILImage()(img)
        mask_pil = transforms.ToPILImage()(mask)

        # 5. 训练时的数据增强
        if self.transform and self.is_train:
            # 同步增强图像和掩码（保证翻转/裁剪一致）
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)
            img_pil = self.transform(img_pil)
            torch.manual_seed(seed)
            mask_pil = self.transform(mask_pil)

        # 6. 转为Tensor并归一化到[0,1]
        img = transforms.ToTensor()(img_pil)  # [3, H, W]，自动归一化到[0,1]
        mask = transforms.ToTensor()(mask_pil)  # [1, H, W]，自动归一化到[0,1]

        # 返回值：图像、掩码、原始尺寸、原始掩码、原始图像
        return img, mask, (ori_h, ori_w), mask_ori, img_ori

# ---------------------- 指标计算（无需修改，兼容numpy输入） ----------------------
def calculate_cc(pred, gt):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    cc, _ = stats.pearsonr(pred_flat, gt_flat)
    return cc

def calculate_kl_div(pred, gt, eps=1e-8):
    pred = pred / (pred.sum() + eps)
    gt = gt / (gt.sum() + eps)
    pred = np.clip(pred, eps, 1.0)
    gt = np.clip(gt, eps, 1.0)
    kl_div = np.sum(gt * np.log(gt / pred))
    return kl_div

def eval_metrics(preds, gts):
    avg_cc = 0.0
    avg_kl = 0.0
    for pred, gt in zip(preds, gts):
        avg_cc += calculate_cc(pred, gt)
        avg_kl += calculate_kl_div(pred, gt)
    avg_cc /= len(preds)
    avg_kl /= len(preds)
    return avg_cc, avg_kl

# ---------------------- 可视化与保存（新增：恢复原始尺寸） ----------------------
def save_prediction(pred, save_path, ori_size):
    """保存预测显著图（恢复到原始图像尺寸）"""
    # 反归一化到[0,255]
    pred = (pred * 255).astype(np.uint8)
    # 恢复原始尺寸
    pred = cv2.resize(pred, (int(ori_size[1]), int(ori_size[0])))  # (w, h)
    cv2.imwrite(save_path, pred)
    
# ---------------------- 结果测试与可视化 ----------------------
def visualize_predictions(model, dataloader, device, save_dir, model_name, num_batches=16):

    # 创建保存目录
    vis_dir = os.path.join(save_dir, 'prediction_visualizations', model_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Predictions will be saved to: {vis_dir}")
    
    # 确保模型在评估模式
    model.eval()
    
    batch_count = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_count >= num_batches:
                break
                
            # 解包批次数据
            imgs, masks, ori_sizes, mask_oris, img_oris = batch_data
            
            # 确保有数据
            if imgs.size(0) == 0:
                continue
            
            # 取第一个样本
            img_tensor = imgs[0:1].to(device)
            mask_tensor = masks[0:1].to(device)
            ori_size = ori_sizes[0]
            mask_ori = mask_oris[0]
            img_ori = img_oris[0]
            
            # 模型预测
            pred = model(img_tensor)
            
            # 转换到CPU和numpy
            img_np = img_ori.numpy() if torch.is_tensor(img_ori) else img_ori
            mask_np = mask_ori.numpy() if torch.is_tensor(mask_ori) else mask_ori
            pred_np = pred.squeeze().cpu().numpy()
            
            # 确保图像格式正确
            # 原图：如果是RGB且值在[0,255]，转换为[0,1]用于显示
            if img_np.max() > 1.0:
                img_np = img_np.astype(np.float32) / 255.0
                
            # 确保预测显著图在[0,1]范围
            if pred_np.min() < 0 or pred_np.max() > 1:
                    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            
            # 确保真实显著图在[0,1]范围
            if mask_np.max() > 1.0:
                mask_np = mask_np.astype(np.float32) / 255.0
            
            # 调整预测图尺寸到原始图像尺寸（如果需要）
            if pred_np.shape != mask_np.shape:
                import cv2
                pred_np = cv2.resize(pred_np, (mask_np.shape[1], mask_np.shape[0]))

            # 调整原图尺寸到真实显著图尺寸
            if img_np.shape[:2] != mask_np.shape:
                img_np = cv2.resize(img_np, (mask_np.shape[1], mask_np.shape[0]))

            # 创建图形
            fig = plt.figure(figsize=(18, 6))
            
           # 定义参数
            image_width = 0.28  # 每个图像的宽度
            cbar_width = 0.015  # colorbar宽度
            gap = 0.02  # 元素之间的间距
            
            # 计算每个元素的水平起始位置
            x0_img1 = 0.05
            x0_img2 = x0_img1 + image_width + gap
            x0_cbar2 = x0_img2 + image_width + 0.005
            x0_img3 = x0_cbar2 + cbar_width + gap
            x0_cbar3 = x0_img3 + image_width + 0.005
            
            # 垂直位置 - 只考虑图像显示区域，不包括标题
            y0_images = 0.12  # 图像底部位置
            image_height = 0.72  # 图像高度
            
            # 创建图像轴
            ax1 = fig.add_axes([x0_img1, y0_images, image_width, image_height])
            ax2 = fig.add_axes([x0_img2, y0_images, image_width, image_height])
            ax3 = fig.add_axes([x0_img3, y0_images, image_width, image_height])
            
            # 子图1：原图
            ax1.imshow(img_np)
            ax1.set_title('Original Image', fontsize=12)
            ax1.axis('off')
            
            # 子图2：预测显著图
            im_pred = ax2.imshow(pred_np, cmap='gray')
            ax2.set_title(f'{model_name} Prediction', fontsize=12)
            ax2.axis('off')
            
            # 子图3：真实显著图
            im_gt = ax3.imshow(mask_np, cmap='gray')
            ax3.set_title('Ground Truth', fontsize=12)
            ax3.axis('off')
            
            # 添加colorbar
            cax2 = fig.add_axes([x0_cbar2, ax2.get_position().y0, cbar_width, ax2.get_position().height])
            cax3 = fig.add_axes([x0_cbar3, ax3.get_position().y0, cbar_width, ax3.get_position().height])
            fig.colorbar(im_pred, cax=cax2)
            fig.colorbar(im_gt, cax=cax3)
            
            # 隐藏colorbar子图的坐标轴
            cax2.axis('off')
            cax3.axis('off')
            
            # 保存图像
            save_path = os.path.join(vis_dir, f'batch_{batch_idx+1}_sample_1.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Generated prediction to {save_path}")
            
            batch_count += 1
            total_samples += 1

    print(f"All predictions were saved to {vis_dir}.")

# ---------------------- 训练日志可视化函数 ----------------------
def read_tensorboard_logs(log_dir, model_names):
        
    logs_data = {}
    
    for model_name in model_names:
        model_log_dir = os.path.join(log_dir, model_name)
        
        # 查找事件文件
        event_files = []
        for root, dirs, files in os.walk(model_log_dir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_files.append(os.path.join(root, file))
        
        if not event_files:
            print(f"警告: {model_log_dir} 中没有找到TensorBoard事件文件")
            continue
        
        # 使用最新的事件文件
        latest_event_file = max(event_files, key=os.path.getmtime)
        
        # 加载事件文件
        try:
            event_acc = EventAccumulator(latest_event_file)
            event_acc.Reload()
        except Exception as e:
            print(f"错误: 无法读取 {latest_event_file}: {e}")
            continue
        
        # 提取标量数据
        model_data = {
            'train_loss': [],
            'val_loss': [],
            'cc': [],
            'kl': [],
            'epochs': []
        }
        
        # 读取训练损失
        if 'Loss/train' in event_acc.Tags()['scalars']:
            train_loss_events = event_acc.Scalars('Loss/train')
            for event in train_loss_events:
                model_data['train_loss'].append(event.value)
                model_data['epochs'].append(event.step)
        
        # 读取验证损失
        if 'Loss/validation' in event_acc.Tags()['scalars']:
            val_loss_events = event_acc.Scalars('Loss/validation')
            for event in val_loss_events:
                model_data['val_loss'].append(event.value)
        
        # 读取CC指标
        if 'Metric/CC' in event_acc.Tags()['scalars']:
            cc_events = event_acc.Scalars('Metric/CC')
            for event in cc_events:
                model_data['cc'].append(event.value)
        
        # 读取KL指标
        if 'Metric/KL Divergence' in event_acc.Tags()['scalars']:
            kl_events = event_acc.Scalars('Metric/KL Divergence')
            for event in kl_events:
                model_data['kl'].append(event.value)
        
        logs_data[model_name] = model_data
    
    return logs_data

def plot_training_metric(log_dir, metric_name, save_fig_path=None, save_txt_path=None):

    # 模型名称和对应的显示名称
    model_names = ['fcn', 'fcn_enhance', 'unet', 'unet-resnet18', 'mobilenetv3-unet', 'swin-unet']
    display_names = ['FCN', 'FCN-Enhanced', 'UNet', 'UNet-ResNet18', 'MobileNetV3-UNet', 'Swin-UNet']
    
    # 颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # 指标显示名称
    metric_display_names = {
        'train_loss': 'Training Loss',
        'val_loss': 'Validation Loss',
        'cc': 'Correlation Coefficient (CC)',
        'kl': 'KL Divergence'
    }
    
    # Y轴标签
    y_labels = {
        'train_loss': 'Loss',
        'val_loss': 'Loss',
        'cc': 'CC Value',
        'kl': 'KL Value'
    }
    
    # 读取日志数据
    print(f"TensorBoard logs are loading...")
    logs_data = read_tensorboard_logs(log_dir, model_names)
    
    if not logs_data:
        print("错误: 没有读取到任何日志数据")
        return None
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置标题和标签
    title = metric_display_names.get(metric_name, metric_name)
    ylabel = y_labels.get(metric_name, metric_name)
    
    ax.set_title(f'{title} - Model Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # 存储每个模型的最佳值和最终值
    model_summary = []
    
    # 绘制每个模型的曲线
    for model_idx, model_name in enumerate(model_names):
        if model_name in logs_data:
            model_data = logs_data[model_name]
            
            # 确保有数据
            if metric_name in model_data and len(model_data[metric_name]) > 0:
                epochs = model_data['epochs'][:len(model_data[metric_name])]
                values = model_data[metric_name]
                
                # 绘制曲线
                ax.plot(epochs, values, 
                       label=display_names[model_idx],
                       color=colors[model_idx],
                       linestyle=line_styles[0],
                       linewidth=2.5,
                       alpha=0.8)
                
                # 计算最佳值和最终值
                if len(values) > 0:
                    final_value = values[-1]
                    
                    if metric_name in ['train_loss', 'val_loss', 'kl']:
                        # 寻找最小值
                        best_idx = np.argmin(values)
                        best_value = values[best_idx]
                        best_epoch = epochs[best_idx]
                        
                        # 标记最佳点
                        ax.plot(best_epoch, best_value, 'o', 
                               color=colors[model_idx], 
                               markersize=8,
                               markeredgecolor='white',
                               markeredgewidth=1.5)
                    elif metric_name == 'cc':
                        # 寻找最大值
                        best_idx = np.argmax(values)
                        best_value = values[best_idx]
                        best_epoch = epochs[best_idx]
                        
                        # 标记最佳点
                        ax.plot(best_epoch, best_value, 'o', 
                               color=colors[model_idx], 
                               markersize=8,
                               markeredgecolor='white',
                               markeredgewidth=1.5)
                    
                    # 添加到汇总
                    model_summary.append({
                        'model': display_names[model_idx],
                        'best_value': best_value,
                        'best_epoch': best_epoch,
                        'final_value': final_value,
                        'total_epochs': len(values)
                    })
    
    # 添加图例
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 调整坐标轴
    if metric_name == 'cc':
        ax.set_ylim(bottom=0)
    elif metric_name == 'kl':
        # 检查是否需要对数坐标
        y_max = ax.get_ylim()[1]
        if y_max > 100:
            ax.set_yscale('log')
            ax.set_ylabel(f'{ylabel} (log scale)', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图像
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')
        print(f"Curves were saved to {save_fig_path}")
    
    plt.show()
    
    # 保存文本汇总
    if save_txt_path and model_summary:
        _save_summary_to_txt(model_summary, metric_name, save_txt_path)
    
    return model_summary

def _save_summary_to_txt(model_summary, metric_name, save_txt_path):
    
    # 确定最佳值的比较方式
    if metric_name in ['train_loss', 'val_loss', 'kl']:
        best_func = min
        best_label = "Min"
    else:  # cc
        best_func = max
        best_label = "Max"
    
    # 计算全局最佳
    if model_summary:
        global_best = best_func(model_summary, key=lambda x: x['best_value'])
    
    with open(save_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL TRAINING SUMMARY - {metric_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # 表头
        header = f"{'Model':<20} {'Best Value':<12} {'Best Epoch':<12} {'Final Value':<12} {'Total Epochs':<12}\n"
        f.write(header)
        f.write("-"*80 + "\n")
        
        # 数据行
        for model in model_summary:
            line = f"{model['model']:<20} {model['best_value']:<12.4f} {model['best_epoch']:<12} {model['final_value']:<12.4f} {model['total_epochs']:<12}\n"
            f.write(line)
        
        f.write("\n" + "="*80 + "\n")
        
        # 添加全局最佳信息
        if model_summary:
            f.write(f"\nGlobal {best_label} {metric_name}: {global_best['best_value']:.4f} (achieved by {global_best['model']} at epoch {global_best['best_epoch']})\n")
        
        f.write("="*80 + "\n")
    
    print(f"Text summary was saved to: {save_txt_path}")

def plot_all_training_metrics(log_dir, output_dir='./training_analysis'):

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("训练日志可视化分析".center(60))
    print("="*60)
    
    # 四个指标
    metrics = ['train_loss', 'val_loss', 'cc', 'kl']
    
    for metric in metrics:
        print(f"\nPloting {metric}...")
        
        # 图像保存路径
        fig_path = os.path.join(output_dir, f'{metric}_curves.png')
        
        # 文本汇总保存路径
        txt_path = os.path.join(output_dir, f'{metric}_summary.txt')
        
        # 绘制单个指标
        plot_training_metric(log_dir, metric, fig_path, txt_path)
    
    print("\n" + "="*60)
    print("Training logfiles have been analysed.".center(60))
    print(f"All results were saved to: {os.path.abspath(output_dir)}")
    print("="*60)


# ---------------------- 工具函数（无需修改） ----------------------
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    