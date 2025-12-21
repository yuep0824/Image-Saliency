import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import metrics
from scipy import stats

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
    """保存预测显著图（恢复到原始图像尺寸，修复类型错误）"""
    # 反归一化到[0,255]
    pred = (pred * 255).astype(np.uint8)
    # 修复：将ori_size转为纯整数元组（处理tensor/ndarray类型）
    ori_w = int(ori_size[1]) if hasattr(ori_size[1], 'item') else ori_size[1]
    ori_h = int(ori_size[0]) if hasattr(ori_size[0], 'item') else ori_size[0]
    # 恢复原始尺寸（w, h）
    pred = cv2.resize(pred, (ori_w, ori_h))
    cv2.imwrite(save_path, pred)
# ---------------------- 工具函数（无需修改） ----------------------
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')