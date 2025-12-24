import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

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


# ---------------------- 分类分析函数 ----------------------
def save_text_results(results, categories, model_names, save_dir):
    """保存文本结果"""
    text_path = os.path.join(save_dir, 'category_results.txt')
    
    with open(text_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("Models' performance on various categories\n".center(120))
        f.write("="*120 + "\n\n")
        
        # 为每个类别创建表格
        for category in categories:
            f.write(f"Category: {category}\n")
            f.write("-"*80 + "\n")
            f.write(f"{'model':<20} {'CC mean':<12} {'KL mean':<12} {'samples':<10}\n")
            f.write("-"*80 + "\n")
            
            for model_name in model_names:
                cc_mean = results[model_name]['cc_mean_by_category'].get(category, 0)
                kl_mean = results[model_name]['kl_mean_by_category'].get(category, 0)
                sample_count = len(results[model_name]['cc_by_category'].get(category, []))
                
                f.write(f"{model_name:<20} {cc_mean:<12.4f} {kl_mean:<12.4f} {sample_count:<10}\n")
            
            f.write("\n")
        
        f.write("="*120 + "\n")
    
    print(f"Text results were saved to {text_path}.")

def bar_cc_by_category(results, categories, model_names, save_dir):
    """绘制CC指标的横向柱状图"""
    # 准备数据
    n_categories = len(categories)
    n_models = len(model_names)
    
    # 为每个模型在每个类别上的平均CC
    cc_matrix = np.zeros((n_categories, n_models))
    
    for i, category in enumerate(categories):
        for j, model_name in enumerate(model_names):
            cc_matrix[i, j] = results[model_name]['cc_mean_by_category'].get(category, 0)
    
    # 颜色方案
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    # 创建图形
    plt.figure(figsize=(16, max(10, n_categories * 0.6)))
    
    y_pos = np.arange(n_categories)
    bar_height = 0.8 / n_models
    
    for j, model_name in enumerate(model_names):
        plt.barh(y_pos + j*bar_height - (n_models-1)*bar_height/2, 
                cc_matrix[:, j], 
                height=bar_height, 
                color=colors[j], 
                label=model_name,
                edgecolor='black',
                linewidth=0.5)
    
    plt.yticks(y_pos, categories, fontsize=10)
    plt.xlabel('CC (Correlation Coefficient)', fontsize=10)
    plt.title('CC on various categories', fontsize=12, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    plt.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # 为每个柱子添加数值标签
    for i in range(n_categories):
        for j in range(n_models):
            x = cc_matrix[i, j]
            y = y_pos[i] + j*bar_height - (n_models-1)*bar_height/2
            if x > 0.01:  # 只显示有意义的数值
                plt.text(x + 0.005, y, f'{x:.3f}', 
                        va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'cc_by_category.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"CC柱状图已保存到: {save_path}")

def bar_kl_by_category(results, categories, model_names, save_dir):
    """绘制KL指标的横向柱状图"""
    # 准备数据
    n_categories = len(categories)
    n_models = len(model_names)
    
    # 为每个模型在每个类别上的平均KL
    kl_matrix = np.zeros((n_categories, n_models))
    
    for i, category in enumerate(categories):
        for j, model_name in enumerate(model_names):
            kl_matrix[i, j] = results[model_name]['kl_mean_by_category'].get(category, 0)
    
    # 颜色方案
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    # 创建图形
    plt.figure(figsize=(16, max(10, n_categories * 0.6)))
    
    y_pos = np.arange(n_categories)
    bar_height = 0.8 / n_models
    
    for j, model_name in enumerate(model_names):
        plt.barh(y_pos + j*bar_height - (n_models-1)*bar_height/2, 
                kl_matrix[:, j], 
                height=bar_height, 
                color=colors[j], 
                label=model_name,
                edgecolor='black',
                linewidth=0.5)
    
    plt.yticks(y_pos, categories, fontsize=10)
    plt.xlabel('KL Divergence', fontsize=10)
    plt.title('KL Divergence on various categories.', fontsize=12, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    plt.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # 为每个柱子添加数值标签
    for i in range(n_categories):
        for j in range(n_models):
            x = kl_matrix[i, j]
            y = y_pos[i] + j*bar_height - (n_models-1)*bar_height/2
            if x > 0.01:  # 只显示有意义的数值
                plt.text(x + 0.005, y, f'{x:.3f}', 
                        va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'kl_by_category.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"KL柱状图已保存到: {save_path}")

def plot_cc_by_category(results, categories, model_names, save_dir):
    """绘制CC指标的折线图（替代横向柱状图）"""
    # 准备数据
    n_categories = len(categories)
    n_models = len(model_names)
    
    # 为每个模型在每个类别上的平均CC
    cc_matrix = np.zeros((n_categories, n_models))
    
    for i, category in enumerate(categories):
        for j, model_name in enumerate(model_names):
            cc_matrix[i, j] = results[model_name]['cc_mean_by_category'].get(category, 0)
    
    # 颜色方案 - 使用Set2颜色映射
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    # 线型方案 - 不同的线型
    line_styles = ['-', '--', '-.', ':', '-.', ':']
    
    # 模型显示名称（简写）
    display_names = ['FCN', 'FCN-E', 'UNet', 'UNet-R18', 'MBV3-UNet', 'Swin-UNet']
    
    # 创建图形
    plt.figure(figsize=(16, 8))
    
    # 绘制每个模型的折线
    for j, (model_name, display_name) in enumerate(zip(model_names, display_names)):
        x_pos = np.arange(n_categories)  # 类别位置
        y_values = cc_matrix[:, j]  # 当前模型的CC值
        
        # 绘制折线，标记点为圆点
        plt.plot(x_pos, y_values, 
                label=display_name,
                color=colors[j],
                linestyle=line_styles[j],
                linewidth=2.5,
                marker='o',  # 标记点为圆点
                markersize=6,
                markerfacecolor=colors[j],
                markeredgecolor='white',
                markeredgewidth=1.5,
                alpha=0.8)
    
    # 设置横轴标签
    plt.xticks(np.arange(n_categories), categories, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 设置轴标签
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('CC (Correlation Coefficient)', fontsize=12)
    plt.title('CC Performance Across Categories', fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, max(0.8, np.max(cc_matrix) * 1.1))  # 设置Y轴范围，留出顶部空间
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9, ncol=2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'cc_by_category_line.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"CC折线图已保存到: {save_path}")


def plot_kl_by_category(results, categories, model_names, save_dir):
    """绘制KL指标的折线图（替代横向柱状图）"""
    # 准备数据
    n_categories = len(categories)
    n_models = len(model_names)
    
    # 为每个模型在每个类别上的平均KL
    kl_matrix = np.zeros((n_categories, n_models))
    
    for i, category in enumerate(categories):
        for j, model_name in enumerate(model_names):
            kl_matrix[i, j] = results[model_name]['kl_mean_by_category'].get(category, 0)
    
    # 颜色方案 - 使用Set1颜色映射
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    # 线型方案 - 不同的线型
    line_styles = ['-', '--', '-.', ':', '-.', ':']
    
    # 模型显示名称（简写）
    display_names = ['FCN', 'FCN-E', 'UNet', 'UNet-R18', 'MBV3-UNet', 'Swin-UNet']
    
    # 创建图形
    plt.figure(figsize=(16, 8))
    
    # 绘制每个模型的折线
    for j, (model_name, display_name) in enumerate(zip(model_names, display_names)):
        x_pos = np.arange(n_categories)  # 类别位置
        y_values = kl_matrix[:, j]  # 当前模型的KL值
        
        # 绘制折线，标记点为圆点
        plt.plot(x_pos, y_values, 
                label=display_name,
                color=colors[j],
                linestyle=line_styles[j],
                linewidth=2.5,
                marker='o',  # 标记点为圆点
                markersize=6,
                markerfacecolor=colors[j],
                markeredgecolor='white',
                markeredgewidth=1.5,
                alpha=0.8)
    
    # 设置横轴标签
    plt.xticks(np.arange(n_categories), categories, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # 设置轴标签
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('KL Divergence', fontsize=12)
    plt.title('KL Divergence Performance Across Categories', fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 检查是否需要对数坐标轴（如果KL值差异很大）
    y_max = np.max(kl_matrix)
    if y_max > 10:  # 如果最大值超过10，使用对数坐标
        plt.yscale('log')
        plt.ylabel('KL Divergence (log scale)', fontsize=12)
        plt.ylim(bottom=0.01)  # 设置对数坐标的最小值
    else:
        plt.ylim(0, y_max * 1.1)  # 线性坐标，留出顶部空间
    
    # 添加图例
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'kl_by_category_line.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"KL折线图已保存到: {save_path}")

def save_overall_stats(results, categories, model_names, save_dir):
    """保存整体统计信息"""
    stats_path = os.path.join(save_dir, 'overall_statistics.txt')
    
    with open(stats_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("整体性能统计\n".center(100))
        f.write("="*100 + "\n\n")
        
        # 计算每个模型的整体平均
        f.write("各模型在所有类别上的平均性能:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'模型':<20} {'整体平均CC':<15} {'整体平均KL':<15} {'最佳类别':<20} {'最差类别':<20}\n")
        f.write("-"*80 + "\n")
        
        for model_name in model_names:
            # 计算整体平均
            all_cc = []
            all_kl = []
            for category in categories:
                cc_values = results[model_name]['cc_by_category'].get(category, [])
                kl_values = results[model_name]['kl_by_category'].get(category, [])
                all_cc.extend(cc_values)
                all_kl.extend(kl_values)
            
            if all_cc:
                overall_cc = np.mean(all_cc)
                overall_kl = np.mean(all_kl)
            else:
                overall_cc = 0
                overall_kl = 0
            
            # 找到最佳和最差类别（按CC）
            category_cc_means = []
            for category in categories:
                cc_mean = results[model_name]['cc_mean_by_category'].get(category, 0)
                category_cc_means.append((category, cc_mean))
            
            if category_cc_means:
                best_category = max(category_cc_means, key=lambda x: x[1])[0]
                worst_category = min(category_cc_means, key=lambda x: x[1])[0]
            else:
                best_category = "N/A"
                worst_category = "N/A"
            
            f.write(f"{model_name:<20} {overall_cc:<15.4f} {overall_kl:<15.4f} {best_category:<20} {worst_category:<20}\n")
        
        f.write("\n" + "="*100 + "\n")
        
        # 按类别统计
        f.write("\n\n按类别统计最佳模型:\n")
        f.write("="*100 + "\n")
        
        for category in categories:
            f.write(f"\n类别: {category}\n")
            f.write("-"*80 + "\n")
            
            # 收集该类别下所有模型的CC值
            model_cc = []
            for model_name in model_names:
                cc_mean = results[model_name]['cc_mean_by_category'].get(category, 0)
                model_cc.append((model_name, cc_mean))
            
            # 按CC值排序
            model_cc_sorted = sorted(model_cc, key=lambda x: x[1], reverse=True)
            
            for rank, (model_name, cc_mean) in enumerate(model_cc_sorted, 1):
                f.write(f"  第{rank}名: {model_name:<20} CC={cc_mean:.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"整体统计已保存到: {stats_path}")

def analyze_by_category_simple(model_dict, data_root, img_size=(256, 256), device='cuda', save_dir=None):

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 验证集路径
    testset_root = os.path.join(data_root, '3-Saliency-TestSet')
    
    # 获取图像和显著图图的根目录
    stimuli_root = os.path.join(testset_root, 'Stimuli')
    fixation_root = os.path.join(testset_root, 'FIXATIONMAPS')

    # 获取所有类别（子文件夹）
    categories = sorted([d for d in os.listdir(stimuli_root) 
                        if os.path.isdir(os.path.join(stimuli_root, d))])
    
    print(f"{len(categories)} categories were found.")
    
    # 初始化结果字典
    results = {}
    model_names = list(model_dict.keys())
    
    for model_name in model_names:
        results[model_name] = {
            'cc_by_category': {category: [] for category in categories},
            'kl_by_category': {category: [] for category in categories},
            'cc_mean_by_category': {},
            'kl_mean_by_category': {}
        }
    
    # 设置模型为评估模式
    for model in model_dict.values():
        model.eval()
    
    # 遍历每个类别
    for category in tqdm(categories, desc="categories"):
        category_stimuli_dir = os.path.join(stimuli_root, category)
        category_fixation_dir = os.path.join(fixation_root, category)
        
        # 获取该类别下的所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            image_files.extend([f for f in os.listdir(category_stimuli_dir) 
                              if f.lower().endswith(ext)])
        
        print(f"\n{category}: {len(image_files)} images")
        
        # 处理每个图像
        for img_file in tqdm(image_files, desc=f"processing {category}", leave=False):
            # 构建完整路径
            img_path = os.path.join(category_stimuli_dir, img_file)
            
            # 查找对应的显著图
            mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")            
            if not os.path.exists(mask_path):
                print(f"Warning: the fixationmap of {img_file} is not exist.")
                continue
            
            # 加载图像和显著图
            img = cv2.imread(img_path)            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            
            mask = cv2.imread(mask_path, 0)  # 灰度图            
            mask = cv2.resize(mask, img_size)
            
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            mask_np = mask.astype(np.float32) / 255.0
            
            # 确保mask在[0,1]范围内
            if mask_np.max() > 1.0:
                mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
            
            # 对每个模型进行预测
            for model_name, model in model_dict.items():
                with torch.no_grad():
                    pred = model(img_tensor)
                    pred_np = pred.squeeze().cpu().numpy()
                    
                    # 确保预测显著图在[0,1]范围
                    if pred_np.min() < 0 or pred_np.max() > 1:
                        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                    
                    # 调整预测图尺寸到与mask相同（如果不同）
                    if pred_np.shape != mask_np.shape:
                        pred_np = cv2.resize(pred_np, (mask_np.shape[1], mask_np.shape[0]))
                    
                    # 计算CC和KL
                    cc = calculate_cc(pred_np, mask_np)
                    kl = calculate_kl_div(pred_np, mask_np)
                    
                    # 保存结果
                    results[model_name]['cc_by_category'][category].append(cc)
                    results[model_name]['kl_by_category'][category].append(kl)

        # 计算每个模型在当前类别的平均指标
        for model_name in model_names:
            if results[model_name]['cc_by_category'][category]:
                cc_values = results[model_name]['cc_by_category'][category]
                kl_values = results[model_name]['kl_by_category'][category]
                
                results[model_name]['cc_mean_by_category'][category] = np.mean(cc_values)
                results[model_name]['kl_mean_by_category'][category] = np.mean(kl_values)
            else:
                results[model_name]['cc_mean_by_category'][category] = 0
                results[model_name]['kl_mean_by_category'][category] = 0
    
    # 保存结果到文本文件
    save_text_results(results, categories, model_names, save_dir)
    
    # 分别绘制CC和KL的横向柱状图
    bar_cc_by_category(results, categories, model_names, save_dir)
    bar_kl_by_category(results, categories, model_names, save_dir)
    plot_cc_by_category(results, categories, model_names, save_dir)
    plot_kl_by_category(results, categories, model_names, save_dir)
    
    # 计算并保存整体统计
    save_overall_stats(results, categories, model_names, save_dir)
    
    print(f"\nAnalysis by categories were completed! Results were saved to {save_dir}")
    
    return results


# ---------------------- 工具函数（无需修改） ----------------------
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    