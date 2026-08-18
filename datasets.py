import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def random_crop(img, crop_size=(80, 80, 80)):
    # img: [c, d, h, w]
    _, d, h, w = img.shape
    pd = np.random.randint(d - crop_size[0] + 1)
    ph = np.random.randint(h - crop_size[1] + 1)
    pw = np.random.randint(w - crop_size[2] + 1)
    patch = img[
        :, pd : pd + crop_size[0], ph : ph + crop_size[1], pw : pw + crop_size[2]
    ]
    return patch


def random_flip(data, p=0.5):
    # data: [C, D, H, W]
    if random.random() < p:
        data = np.flip(data, axis=1)
    if random.random() < p:
        data = np.flip(data, axis=2)
    if random.random() < p:
        data = np.flip(data, axis=3)
    return data


def random_scale_one_channel(data, p=0.5):
    if random.random() < p:
        scale = random.uniform(0.9, 1.1)
        data = data * scale
    return data


def random_scale(data, c=4, p=0.5):
    for i in range(c):
        data[i] = random_scale_one_channel(data[i], p)
    return data


class BraTS(Dataset):
    def __init__(
        self,
        base_dir: str = "../data",
        mode: str = "train",             # 支持 'train', 'val', 或 'test'
        crop_size=(80, 80, 80),
        flip: bool = True,
        scale: bool = True,
        return_id: bool = False,
        ensure_fg: bool = False,
        fg_min_ratio: float = 0.0,
        ensure_et_min_ratio: float = 0.0,
        fg_max_attempts: int = 8,
        missing_modalities: list = None  # 例如传入 [1, 2] 会自动把 T2 和 T1ce 模态置零
    ):
        self.mode = mode
        self.crop_size = crop_size
        
        # 验证和测试模式下，强制关闭所有随机数据增强
        self.flip = flip if mode == "train" else False
        self.scale = scale if mode == "train" else False
        self.ensure_fg = ensure_fg if mode == "train" else False
        
        self.return_id = return_id
        self.fg_min_ratio = float(fg_min_ratio)
        self.ensure_et_min_ratio = float(ensure_et_min_ratio)
        self.fg_max_attempts = int(fg_max_attempts)
        self.missing_modalities = missing_modalities if missing_modalities is not None else []

        imglist = []
        # 根据 mode 动态加载对应的 txt 列表
        list_path = os.path.join(base_dir, f"{mode}_list.txt")
        with open(list_path, "r") as f:
            lines = f.readlines()
            for ll in lines:
                imglist.append(ll.strip())
                
        # 适配咱们刚刚生成的 brats2020 文件夹结构
        self.imglist = [os.path.join(base_dir, "brats2020", f"{x}.npy") for x in imglist]

    def _random_crop_with_fg(self, data: np.ndarray) -> np.ndarray:
        """
        多次尝试随机裁剪，直到返回一个包含足够肿瘤前景的 Patch。
        """
        if not self.ensure_fg or self.fg_min_ratio <= 0.0:
            return random_crop(data, self.crop_size)

        attempts = max(1, self.fg_max_attempts)
        best_patch = None
        best_ratio = -1.0
        best_et_ratio = -1.0
        
        for _ in range(attempts):
            patch = random_crop(data, self.crop_size)
            label_patch = patch[4]
            total = label_patch.size
            
            # 前景包含 1, 2, 3 (增强肿瘤、坏死、水肿)
            fg = np.count_nonzero(label_patch > 0)
            ratio = float(fg) / float(total) if total > 0 else 0.0
            
            # 注意：在我们的 preprocess.py 中，原始的 4(增强肿瘤) 被映射为了 1。
            # 因此这里用 label_patch == 1 来准确统计增强肿瘤(ET)的比例。
            et = np.count_nonzero(label_patch == 1)
            et_ratio = float(et) / float(total) if total > 0 else 0.0
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_patch = patch
                best_et_ratio = et_ratio
            elif et_ratio > best_et_ratio:
                best_et_ratio = et_ratio
                best_patch = patch
                
            if ratio >= self.fg_min_ratio:
                if self.ensure_et_min_ratio > 0.0:
                    if et_ratio >= self.ensure_et_min_ratio:
                        return patch
                else:
                    return patch
                    
        return best_patch if best_patch is not None else random_crop(data, self.crop_size)

    def __getitem__(self, index):
        path = self.imglist[index]
        data = np.load(path)
        
        # 1. 裁剪策略
        if self.mode == "train":
            data = self._random_crop_with_fg(data)
        else:
            # 验证/测试时执行固定的中心裁剪 (Center Crop)，保证每次评估输入完全一致
            _, d, h, w = data.shape
            pd = (d - self.crop_size[0]) // 2
            ph = (h - self.crop_size[1]) // 2
            pw = (w - self.crop_size[2]) // 2
            data = data[:, pd:pd+self.crop_size[0], ph:ph+self.crop_size[1], pw:pw+self.crop_size[2]]

        # 2. 空间数据增强 (仅训练时)
        if self.flip:
            data = random_flip(data)

        image = data[0:4].copy()
        label = data[4:].copy() 

        # 3. 动态模拟模态缺失
        # 如果传入了 [1] (代表缺失 T2)，则将第 1 通道全部置零
        for m in self.missing_modalities:
            if 0 <= m < 4:
                image[m] = 0.0

        # 4. 亮度数据增强 (仅训练时)
        if self.scale:
            image = random_scale(image)

        # 转换为 PyTorch Tensor
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))

        if self.return_id:
            case_id = os.path.basename(path).replace('.npy', '')
            return image, label, case_id
            
        return image, label

    def __len__(self):
        return len(self.imglist)