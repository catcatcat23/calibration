import torch
import numpy as np
import random
import pandas as pd
import os
from torchvision.io import read_image
from utils.utilities import MaskOutROI
import torchvision.transforms.functional as TF
import random
from typing import Sequence
from scipy.special import softmax
from dataloader.image_resampler import DiscreteRotate
import cv2
from copy import deepcopy
def visualize_img_pair(path_ng, path_ok):

    sNG_image = cv2.imread(path_ng, cv2.IMREAD_COLOR)
    sOK_image = cv2.imread(path_ok, cv2.IMREAD_COLOR)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(5, 5))

    sNG_image_show = cv2.cvtColor(sNG_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(sNG_image_show, cmap='gray')
    axes[0].set_title(f'NG')

    sNG_image_show = cv2.cvtColor(sOK_image, cv2.COLOR_BGR2RGB)
    axes[1].imshow(sNG_image_show, cmap='gray')
    axes[1].set_title(f'OK')
    plt.tight_layout()
    plt.show()

class LUT:
    def __init__(self, lut_path, lut_p):
        self.lut_img = cv2.imread(lut_path, cv2.IMREAD_COLOR)
        self.lut_p = lut_p

    def __call__(self, x):
        c, h, w = x.shape
        if c == 1:
            lut_x_tensor = x
        else:
            if np.random.rand() <= self.lut_p:
                x_np = (x*255).numpy().astype(np.uint8).transpose(1, 2, 0)
                lut_x_np = cv2.LUT(x_np, self.lut_img)
                lut_x_tensor = torch.tensor(lut_x_np.transpose(2, 0, 1)) / 255
            else:
                lut_x_tensor = x
        return lut_x_tensor

class LUT_VAL:
    def __init__(self, lut_path, lut_p):
        self.lut_img = cv2.imread(lut_path, cv2.IMREAD_COLOR)
        self.lut_p = lut_p

    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        h, w, c = x.shape
        if c == 1:
            lut_x_tensor = x
        else:
            if np.random.rand() <= self.lut_p:
                # x_np = (x*255).numpy().astype(np.uint8).transpose(1, 2, 0)
                lut_x_tensor = cv2.LUT(x, self.lut_img)
                # lut_x_tensor = torch.tensor(lut_x_np.transpose(2, 0, 1)) / 255
            else:
                lut_x_tensor = x
        return lut_x_tensor

class LUT_DEBUG:
    def __init__(self, lut_path, lut_p):
        self.lut_img = cv2.imread(lut_path, cv2.IMREAD_COLOR)
        self.lut_p = lut_p

    def __call__(self, x, save_img_path = None):
        c, h, w = x.shape
        if c == 1:
            lut_x_tensor = x
        else:
            if np.random.rand() <= self.lut_p:
                # x_np = (x*255).astype(np.uint8).transpose(1, 2, 0)
                lut_x_np = cv2.LUT(x, self.lut_img)
                cv2.imwrite(save_img_path, lut_x_np)
                lut_x_tensor = torch.tensor(lut_x_np.transpose(2, 0, 1)) / 255
            else:
                lut_x_tensor = x
        return lut_x_tensor

def visual_tensor_img(s1_X):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(s1_X.permute(1, 2, 0))
    plt.title('s1')
    plt.show()

from PIL import Image
from utils.utilities import compress_img, compress_img_opencv, compress_img_PIL

class ImageLoader3(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path_pairs, label_pairs, binary_label_pairs, 
                 positions, compression_p = 0, p_range = [65, 95], select_p=None, transform=None, transform_same=None, transform_sync=None):
        self.root_folder = root_folder
        self.img_path_pairs = img_path_pairs
        self.label_pairs = label_pairs
        self.binary_label_pairs = binary_label_pairs
        self.positions = positions
        self.compression_p = compression_p
        self.p_range = p_range
        self.transform = transform
        self.transform_sync  = transform_sync
        self.transform_same = transform_same
        self.seed = 0
        if select_p == 1:
            self.comp_func = compress_img_PIL
        elif select_p == 0:
            self.comp_func = compress_img_opencv
        else:
            raise ValueError(f'只能选择一个压缩方式')
    def __getitem__(self, index):
        img1_path, img2_path = self.img_path_pairs[index]
        s1_y, s2_y = self.label_pairs[index]  # 缺陷标签
        binary_y = self.binary_label_pairs[index]
        position = self.positions[index]
        s1_img_path = os.path.join(self.root_folder, img1_path)
        s2_img_path = os.path.join(self.root_folder, img2_path)

        try:
            if random.random() < self.compression_p:
                random.seed(self.seed)
                s1_X_img =  Image.open(s1_img_path)
                s2_X_img =  Image.open(s2_img_path)

                p = random.randint(self.p_range[0], self.p_range[1])
                s1_X = self.comp_func(s1_X_img, p) / 255
                s2_X = self.comp_func(s2_X_img, p) / 255
            else: 
                # read_image(os.path.join(self.root_folder, img1_path)) = np.array(Image.open(s1_img_path)).transpose(2,0,1)
                s1_X = read_image(s1_img_path) / 255
                s2_X = read_image(s2_img_path) / 255
                if s1_X.shape[0] == 4 or s2_X.shape[0] == 4:
                    s1_X = s1_X[:3]
                    s2_X = s2_X[:3] 
        except:
            print(img1_path)
            print(img2_path)
            print(s1_y, s2_y)
            raise ValueError(f'数据读取有误')

        if img1_path == img2_path:

            if self.transform_same is not None:
                s1_X = self.transform_same(s1_X)
                s2_X = self.transform_same(s2_X)
            else:
                s1_X = self.transform(s1_X)
                s2_X = self.transform(s2_X)

        else: # 非同步的，颜色抖动和随机裁剪可能不一样

            s1_X = self.transform(s1_X)
            s2_X = self.transform(s2_X)

        if self.transform_sync is not None:  #同步的变换，翻转，旋转啥的都一致
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s1_X = self.transform_sync(s1_X)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s2_X = self.transform_sync(s2_X)  
            self.seed += 1

        return s1_X, s2_X, s1_y, s2_y, binary_y #, position

    def __len__(self):
        return len(self.label_pairs)

def diff_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img, return_PIL = False):
    S1_diff = s1_X - s1_match_X_img
    S2_diff = s2_X - s2_match_X_img

    s1_X = S1_diff - S1_diff.min()
    s2_X = S2_diff - S2_diff.min()

    s1_X = s1_X / 510 # 255 * 2
    s2_X = s2_X / 510
    if return_PIL:
        s1_X = Image.fromarray((s1_X * 255).astype(np.uint8))
        s2_X = Image.fromarray((s2_X * 255).astype(np.uint8))

    return s1_X, s2_X

def diff_rgb_white_abs(s1_X, s2_X, s1_match_X_img, s2_match_X_img, return_PIL = False):
    S1_diff = abs(s1_X - s1_match_X_img)
    S2_diff = abs(s2_X - s2_match_X_img)

    # s1_X_img = to_pil_image((S1_diff).to(torch.uint8))
    # s2_X_img = to_pil_image((S2_diff).to(torch.uint8))
    # s1_X_img.save(save_img1_path)
    # s2_X_img.save(save_img2_path)

    if return_PIL:
        s1_X = Image.fromarray((S1_diff).astype(np.uint8))
        s2_X = Image.fromarray((S2_diff).astype(np.uint8))
    else:
        s1_X = S1_diff / 255 # 255 * 2
        s2_X = S2_diff / 255
    return s1_X, s2_X


def max_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img, return_PIL = False):
    if isinstance(s1_X, torch.Tensor):
        s1_X = torch.maximum(s1_X, s1_match_X_img)
        s2_X = torch.maximum(s2_X, s2_match_X_img)
    elif isinstance(s1_X, np.ndarray):
        s1_X = np.maximum(s1_X, s1_match_X_img)
        s2_X = np.maximum(s2_X, s2_match_X_img)

    if return_PIL:
        s1_X = Image.fromarray(s1_X.astype(np.uint8))
        s2_X = Image.fromarray(s2_X.astype(np.uint8))
    else:
        s1_X = s1_X / 255
        s2_X = s2_X / 255

    return s1_X, s2_X

def add_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img, return_PIL = False):
    s1_X = s1_X + s1_match_X_img
    s2_X = s2_X + s2_match_X_img

    s1_X = s1_X / 510
    s2_X = s2_X / 510
    if return_PIL:
        s1_X = Image.fromarray((s1_X * 255).astype(np.uint8))
        s2_X = Image.fromarray((s2_X * 255).astype(np.uint8))

    return s1_X, s2_X

def cat_channel_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img):
    if isinstance(s1_X, torch.Tensor):
        s1_X = torch.cat([s1_X, s1_match_X_img], dim=0)
        s2_X = torch.cat([s2_X, s2_match_X_img], dim=0)
        return s1_X, s2_X

from torchvision.transforms.functional import to_pil_image
def cat_HW_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img, dim, cat_direction = 'v', norm = False, return_PIL=True):

    if cat_direction == 'v':
        if isinstance(s1_X, np.ndarray):
            s1_X = np.concatenate([s1_X, s1_match_X_img], axis=dim)
            s2_X = np.concatenate([s2_X, s2_match_X_img], axis=dim)

        elif isinstance(s1_X, torch.Tensor):
            s1_X = torch.cat([s1_X, s1_match_X_img], axis=dim)
            s2_X = torch.cat([s2_X, s2_match_X_img], axis=dim)

    elif cat_direction == 'h':
        if isinstance(s1_X, np.ndarray):
            s1_X = np.concatenate([s1_X, s1_match_X_img], axis=dim)
            s2_X = np.concatenate([s2_X, s2_match_X_img], axis=dim)
        elif isinstance(s1_X, torch.Tensor):
            s1_X = torch.cat([s1_X, s1_match_X_img], axis=dim)
            s2_X = torch.cat([s2_X, s2_match_X_img], axis=dim)
    
    if norm:
        s1_X = s1_X/ 255
        s2_X = s2_X/ 255

        return s1_X, s2_X

    if return_PIL:
        if isinstance(s1_X, np.ndarray):
            s1_X = Image.fromarray((s1_X).astype(np.uint8))
            s2_X = Image.fromarray((s2_X).astype(np.uint8))
        elif isinstance(s1_X, torch.Tensor):
            s1_X = to_pil_image((s1_X).to(torch.uint8))
            s2_X = to_pil_image((s2_X).to(torch.uint8))

    return s1_X, s2_X

def norm_np2PIL(s1_X_img):
    s1_X_img = Image.fromarray((s1_X_img * 255).astype(np.uint8))
    return s1_X_img

from torchvision.io.image import ImageReadMode
class ImageFusionLoader(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path_pairs, label_pairs, binary_label_pairs, 
                 positions, compression_p = 0, p_range = [65, 95], select_p=None, transform=None, transform_gray = None, transform_same=None, transform_sync=None, transform_sync_gray=None, 
                 fusion_type = 'cat'):
        self.root_folder = root_folder
        self.img_path_pairs = img_path_pairs
        self.label_pairs = label_pairs
        self.binary_label_pairs = binary_label_pairs
        self.positions = positions
        self.compression_p = compression_p
        self.p_range = p_range
        self.select_p = select_p
        self.transform = transform
        self.transform_gray = transform_gray
        self.transform_sync  = transform_sync
        self.transform_sync_gray = transform_sync_gray
        self.transform_same = transform_same
        self.seed = 0

        self.fusion_type = fusion_type
        self._fusion_rgb_white()

    def _fusion_rgb_white(self):
        del_idx = []
        for idx, pair_path in enumerate(self.img_path_pairs):
            img1_path, img2_path = self.img_path_pairs[idx]

            if '_rgb_' in img1_path:
                s1_match_path = os.path.join(self.root_folder, img1_path.replace('_rgb_', '_white_'))
                s2_match_path = os.path.join(self.root_folder, img2_path.replace('_rgb_', '_white_'))

                if not os.path.exists(s1_match_path) or not os.path.exists(s2_match_path):
                    del_idx.append(idx)
            elif '_RGB_' in img1_path:
                s1_match_path = os.path.join(self.root_folder, img1_path.replace('_RGB_', '_WHITE_'))
                s2_match_path = os.path.join(self.root_folder, img2_path.replace('_RGB_', '_WHITE_'))

                if not os.path.exists(s1_match_path) or not os.path.exists(s2_match_path):
                    del_idx.append(idx)
                
            elif '_white_' in img1_path:
                s1_match_path = os.path.join(self.root_folder, img1_path.replace('_white_', '_rgb_'))
                s2_match_path = os.path.join(self.root_folder, img2_path.replace('_white_', '_rgb_'))

                if not os.path.exists(s1_match_path) or not os.path.exists(s2_match_path):
                    del_idx.append(idx)
            elif '_WHITE_' in img1_path:
                s1_match_path = os.path.join(self.root_folder, img1_path.replace('_WHITE_', '_RGB_'))
                s2_match_path = os.path.join(self.root_folder, img2_path.replace('_WHITE_', '_RGB_'))

                if not os.path.exists(s1_match_path) or not os.path.exists(s2_match_path):
                    del_idx.append(idx)
            else:
                # 没有rgb标识的要去掉
                del_idx.append(idx)


        self.img_path_pairs = [v for idx, v in enumerate(self.img_path_pairs) if idx not in del_idx]
        self.label_pairs = [v for idx, v in enumerate(self.label_pairs) if idx not in del_idx]
        self.positions = [v for idx, v in enumerate(self.positions) if idx not in del_idx]
        self.binary_label_pairs = [v for idx, v in enumerate(self.binary_label_pairs) if idx not in del_idx]

    def __getitem__(self, index):
        img1_path, img2_path = self.img_path_pairs[index]
        s1_y, s2_y = self.label_pairs[index]  # 缺陷标签
        binary_y = self.binary_label_pairs[index]
        position = self.positions[index]
        s1_img_path = os.path.join(self.root_folder, img1_path)
        s2_img_path = os.path.join(self.root_folder, img2_path)

        # for debug
        # img1_name = img1_path.split('/')[-1]
        # img2_name = img2_path.split('/')[-1]

        # vis_dir = f'/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/vis_rgbwhite/{self.fusion_type}'
        # os.makedirs(vis_dir, exist_ok=True)
        # save_img1_path = os.path.join(vis_dir, img1_name.replace('.png',f"_ref_{binary_y}_{self.fusion_type}.jpg"))
        # save_img2_path = os.path.join(vis_dir, img2_name.replace('.png',f"_insp_{binary_y}_{self.fusion_type}.jpg"))

        if '_rgb_' in img1_path or '_RGB_' in img1_path:
            if '_rgb_' in img1_path:
                s1_match_img_path = os.path.join(self.root_folder, img1_path.replace('_rgb_', '_white_'))
                s2_match_img_path = os.path.join(self.root_folder, img2_path.replace('_rgb_', '_white_'))
            elif '_RGB_' in img1_path:
                s1_match_img_path = os.path.join(self.root_folder, img1_path.replace('_RGB_', '_WHITE_'))
                s2_match_img_path = os.path.join(self.root_folder, img2_path.replace('_RGB_', '_WHITE_'))

            if self.fusion_type == 'diff_white_rgb' or self.fusion_type == 'cat_HW_white_rgb':
                s1_img_path_tp = s1_img_path
                s2_img_path_tp = s2_img_path
                s1_img_path = s1_match_img_path
                s2_img_path = s2_match_img_path
                s1_match_img_path = s1_img_path_tp
                s2_match_img_path = s2_img_path_tp

        elif '_white_' in img2_path or '_WHITE_' in img2_path:
            if '_white_' in img2_path:
                s1_match_img_path = os.path.join(self.root_folder, img1_path.replace('_white_', '_rgb_'))
                s2_match_img_path = os.path.join(self.root_folder, img2_path.replace('_white_', '_rgb_'))
            elif '_WHITE_' in img2_path:
                s1_match_img_path = os.path.join(self.root_folder, img1_path.replace('_WHITE_', '_RGB_'))
                s2_match_img_path = os.path.join(self.root_folder, img2_path.replace('_WHITE_', '_RGB_'))

            if self.fusion_type == 'diff_rgb_white' or self.fusion_type == 'cat_HW_rgb_white':
                s1_img_path_tp = s1_img_path
                s2_img_path_tp = s2_img_path
                s1_img_path = s1_match_img_path
                s2_img_path = s2_match_img_path
                s1_match_img_path = s1_img_path_tp
                s2_match_img_path = s2_img_path_tp

        try:
            if random.random() < self.compression_p:
                random.seed(self.seed)
                s1_X_img =  np.array(Image.open(s1_img_path)).astype(np.float32)
                s2_X_img =  np.array(Image.open(s2_img_path)).astype(np.float32)

                s1_match_X_img =  np.array(Image.open(s1_match_img_path)).astype(np.float32)
                s2_match_X_img =  np.array(Image.open(s2_match_img_path)).astype(np.float32)

                if self.fusion_type == 'diff_rgb_white' or self.fusion_type == 'diff_white_rgb':
                    s1_X_img, s2_X_img = diff_rgb_white(s1_X_img, s2_X_img, s1_match_X_img, s2_match_X_img, return_PIL=True)
                
                elif self.fusion_type == 'diff_rgb_white_abs':   
                    s1_X_img, s2_X_img = diff_rgb_white_abs(s1_X_img, s2_X_img, s1_match_X_img, s2_match_X_img, return_PIL=True)

                elif self.fusion_type == 'max':
                    s1_X_img, s2_X_img = max_rgb_white(s1_X_img, s2_X_img, s1_match_X_img, s2_match_X_img, return_PIL=True)

                elif self.fusion_type == 'add':
                    s1_X_img, s2_X_img = add_rgb_white(s1_X_img, s2_X_img, s1_match_X_img, s2_match_X_img, return_PIL=True)

                elif 'cat_HW' in self.fusion_type:
                    s1_X_img, s2_X_img = cat_HW_rgb_white(s1_X_img, s2_X_img, s1_match_X_img, s2_match_X_img, dim = 0, cat_direction = 'v', return_PIL=True)

                else:
                    s1_X_img = Image.fromarray(s1_X_img.astype(np.uint8))
                    s2_X_img = Image.fromarray(s2_X_img.astype(np.uint8))

                # for debug    
                # s1_X_img.save(save_img1_path)
                # s2_X_img.save(save_img2_path)

                if self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
                    if '_rgb_' in s1_img_path or '_RGB_' in s1_img_path:
                        s1_match_X_img = Image.fromarray(s1_match_X_img.astype(np.uint8)).convert('L')
                        s2_match_X_img = Image.fromarray(s2_match_X_img.astype(np.uint8)).convert('L')
                    else:
                        s1_match_X_img_ = deepcopy(s1_match_X_img)  # RGB图
                        s2_match_X_img_ = deepcopy(s2_match_X_img)

                        s1_match_X_img = s1_X_img.convert('L') # 白图
                        s2_match_X_img = s2_X_img.convert('L')

                        s1_X_img =  Image.fromarray(s1_match_X_img_.astype(np.uint8))
                        s2_X_img =  Image.fromarray(s2_match_X_img_.astype(np.uint8))

                else:
                    s1_match_X_img = Image.fromarray(s1_match_X_img.astype(np.uint8))
                    s2_match_X_img = Image.fromarray(s2_match_X_img.astype(np.uint8))

                p = random.randint(self.p_range[0], self.p_range[1])
                select_p = random.random()

                s1_match_X_img = compress_img(s1_match_X_img, p, select_p) / 255
                s2_match_X_img = compress_img(s2_match_X_img, p, select_p) / 255

                s1_X = compress_img(s1_X_img, p, select_p) / 255
                s2_X = compress_img(s2_X_img, p, select_p) / 255

            else: 
                # s1_X = read_image(s1_img_path) / 255 # torch.float32数据类型
                # s2_X = read_image(s2_img_path) / 255

                s1_X = read_image(s1_img_path).to(torch.float32)
                s2_X = read_image(s2_img_path).to(torch.float32)
                if self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
                    if '_rgb_' in s1_img_path or '_RGB_' in s1_img_path:
                        s1_match_X_img =  read_image(s1_match_img_path, mode=ImageReadMode.GRAY).to(torch.float32)
                        s2_match_X_img =  read_image(s2_match_img_path, mode=ImageReadMode.GRAY).to(torch.float32)
                    else:
                        s1_X = read_image(s1_match_img_path).to(torch.float32)
                        s2_X = read_image(s2_match_img_path).to(torch.float32)
                        s1_match_X_img =  read_image(s1_img_path, mode=ImageReadMode.GRAY).to(torch.float32)
                        s2_match_X_img =  read_image(s2_img_path, mode=ImageReadMode.GRAY).to(torch.float32)
                else:
                    s1_match_X_img =  read_image(s1_match_img_path).to(torch.float32)
                    s2_match_X_img =  read_image(s2_match_img_path).to(torch.float32)

                    assert s1_X.shape == s1_match_X_img.shape, f"{img1_path}rgb和白图尺寸不一致"
                    assert s2_X.shape == s2_match_X_img.shape, f"{img2_path}rgb和白图尺寸不一致"

                # for_debug
                # if 'cat_HW' in self.fusion_type:
                #     s1_X_, s2_X_ = cat_HW_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img, dim = 1, cat_direction = 'v', return_PIL=True)
                #     s1_X_.save(save_img1_path)
                #     s2_X_.save(save_img2_path)

                if self.fusion_type == 'diff_rgb_white' or self.fusion_type == 'diff_white_rgb':
                    s1_X, s2_X = diff_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img)
                    
                elif self.fusion_type == 'diff_rgb_white_abs':   
                    s1_X_img, s2_X_img = diff_rgb_white_abs(s1_X, s2_X, s1_match_X_img, s2_match_X_img)

                elif self.fusion_type == 'max':
                    s1_X, s2_X = max_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img)
                elif self.fusion_type == 'add':
                    s1_X, s2_X = add_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img)
                elif 'cat_HW' in self.fusion_type:
                    s1_X, s2_X = cat_HW_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img, dim = 1, cat_direction = 'v', norm=True)
                else:
                    s1_X = s1_X / 255
                    s2_X = s2_X / 255

                    s1_match_X_img = s1_match_X_img / 255
                    s2_match_X_img = s2_match_X_img / 255

        except Exception as e:
            print(img1_path)
            print(img2_path)
            print(s1_y, s2_y)
            print("An error occurred:", str(e))
            raise ValueError(f'数据读取有误')

        if img1_path == img2_path:

            if self.transform_same is not None:
                s1_X = self.transform_same(s1_X)
                s2_X = self.transform_same(s2_X)
                s1_match_X_img = self.transform_same(s1_match_X_img)
                s2_match_X_img = self.transform_same(s2_match_X_img)
            else:
                s1_X = self.transform(s1_X)
                s2_X = self.transform(s2_X)
                # s1_match_X_img = self.transform(s1_match_X_img)
                # s2_match_X_img = self.transform(s2_match_X_img)
        else: # 非同步的，颜色抖动和随机裁剪可能不一样

            s1_X = self.transform(s1_X)
            s2_X = self.transform(s2_X)

        if self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
            s1_match_X_img = self.transform_gray(s1_match_X_img)
            s2_match_X_img = self.transform_gray(s2_match_X_img)
        else:
            s1_match_X_img = self.transform(s1_match_X_img)
            s2_match_X_img = self.transform(s2_match_X_img)

        if self.transform_sync is not None:  #同步的变换，翻转，旋转啥的都一致
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s1_X = self.transform_sync(s1_X)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            if self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
                s1_match_X_img = self.transform_sync_gray(s1_match_X_img)
            else:
                s1_match_X_img = self.transform_sync(s1_match_X_img)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s2_X = self.transform_sync(s2_X)  

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            if self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
                s2_match_X_img = self.transform_sync_gray(s2_match_X_img)
            else:
                s2_match_X_img = self.transform_sync(s2_match_X_img)
            # s2_match_X_img = self.transform_sync(s2_match_X_img)  

            self.seed += 1

        if self.fusion_type == 'cat' or self.fusion_type == 'cat_gray' or self.fusion_type == 'merge_rb_G':
            if self.fusion_type == 'merge_rb_G':
                s1_X = s1_X[[0,2], :, :]
                s2_X = s2_X[[0,2], :, :]

            s1_X, s2_X = cat_channel_rgb_white(s1_X, s2_X, s1_match_X_img, s2_match_X_img)

        return s1_X, s2_X, s1_y, s2_y, binary_y #, position

    def __len__(self):
        return len(self.label_pairs)


class ImageLoader2(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path_pairs, label_pairs, binary_label_pairs, 
                 positions, compression_p = 0, p_range = [65, 95], select_p=None, transform=None, transform_same=None, transform_sync=None):
        self.root_folder = root_folder
        self.img_path_pairs = img_path_pairs
        self.label_pairs = label_pairs
        self.binary_label_pairs = binary_label_pairs
        self.positions = positions
        self.compression_p = compression_p
        self.p_range = p_range
        self.select_p = select_p
        self.transform = transform
        self.transform_sync  = transform_sync
        self.transform_same = transform_same
        self.seed = 0

    def __getitem__(self, index):
        img1_path, img2_path = self.img_path_pairs[index]
        s1_y, s2_y = self.label_pairs[index]  # 缺陷标签
        binary_y = self.binary_label_pairs[index]
        position = self.positions[index]
        s1_img_path = os.path.join(self.root_folder, img1_path)
        s2_img_path = os.path.join(self.root_folder, img2_path)

        try:
            if random.random() < self.compression_p:
                random.seed(self.seed)
                s1_X_img =  Image.open(s1_img_path)
                s2_X_img =  Image.open(s2_img_path)

                p = random.randint(self.p_range[0], self.p_range[1])
                select_p = random.random()
                s1_X = compress_img(s1_X_img, p, select_p) / 255
                s2_X = compress_img(s2_X_img, p, select_p) / 255
            else: 
                # read_image(os.path.join(self.root_folder, img1_path)) = np.array(Image.open(s1_img_path)).transpose(2,0,1)
                s1_X = read_image(s1_img_path) / 255
                s2_X = read_image(s2_img_path) / 255
                if s1_X.shape[0] == 4 or s2_X.shape[0] == 4:
                    s1_X = s1_X[:3]
                    s2_X = s2_X[:3] 
        except:
            print(img1_path)
            print(img2_path)
            print(s1_y, s2_y)
            raise ValueError(f'数据读取有误')

        if img1_path == img2_path:

            if self.transform_same is not None:
                s1_X = self.transform_same(s1_X)
                s2_X = self.transform_same(s2_X)
            else:
                s1_X = self.transform(s1_X)
                s2_X = self.transform(s2_X)

        else: # 非同步的，颜色抖动和随机裁剪可能不一样

            s1_X = self.transform(s1_X)
            s2_X = self.transform(s2_X)

        if self.transform_sync is not None:  #同步的变换，翻转，旋转啥的都一致
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s1_X = self.transform_sync(s1_X)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s2_X = self.transform_sync(s2_X)  
            self.seed += 1

        return s1_X, s2_X, s1_y, s2_y, binary_y #, position

    def __len__(self):
        return len(self.label_pairs)
    
class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path_pairs, label_pairs, binary_label_pairs, positions, transform=None, transform_same=None, transform_sync=None):
        self.root_folder = root_folder
        self.img_path_pairs = img_path_pairs
        self.label_pairs = label_pairs
        self.binary_label_pairs = binary_label_pairs
        self.positions = positions
        self.transform = transform
        self.transform_sync  = transform_sync
        self.transform_same = transform_same
        self.seed = 0

    def __getitem__(self, index):
        img1_path, img2_path = self.img_path_pairs[index]
        s1_y, s2_y = self.label_pairs[index]  # 缺陷标签
        binary_y = self.binary_label_pairs[index]
        position = self.positions[index]
   
        # if not os.path.exists(os.path.join(self.root_folder, img1_path)) or not os.path.exists(os.path.join(self.root_folder, img1_path)):
        #     print(img1_path)
        #     print(img2_path)
        s1_X = read_image(os.path.join(self.root_folder, img1_path)) / 255
        s2_X = read_image(os.path.join(self.root_folder, img2_path)) / 255
        if s1_X.shape[0] == 4 or s2_X.shape[0] == 4:
            s1_X = s1_X[:3]
            s2_X = s2_X[:3]

        if img1_path == img2_path:

            if self.transform_same is not None:
                s1_X = self.transform_same(s1_X)
                s2_X = self.transform_same(s2_X)
            else:
                s1_X = self.transform(s1_X)
                s2_X = self.transform(s2_X)

        else: # 非同步的，颜色抖动和随机裁剪可能不一样
            s1_X = self.transform(s1_X)
            s2_X = self.transform(s2_X)

        if self.transform_sync is not None:  #同步的变换，翻转，旋转啥的都一致
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s1_X = self.transform_sync(s1_X)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s2_X = self.transform_sync(s2_X)  
            self.seed += 1

        # print(s1_X.shape, s2_X.shape, s1_y, s2_y, binary_y, position, os.path.join(self.root_folder, img1_path))
        # print(s1_X.shape, s2_X.shape, type(s1_y), type(s2_y), type(binary_y), type(position))
        return s1_X, s2_X, s1_y, s2_y, binary_y #, position

    def __len__(self):
        return len(self.label_pairs)

def mp_weighted_resampler(annotation_df, batch_size=64, n_max_pairs=100, k=1.2, seed=42):
    n_total = len(annotation_df)
    material_id_list, material_id_counts = np.unique(annotation_df['material_id'], return_counts=True)
    # # compute the sampling probability for each material subset
    p_material = 1 - material_id_counts / n_total
    p_material_norm = np.array(p_material) / sum(p_material)


    # ng_material_id_list, ng_material_id_counts = np.unique(annotation_df[annotation_df['original_y'] != 0]['material_id'], return_counts=True)
    # k_sample_list = (np.max(ng_material_id_counts)*(1 / ng_material_id_counts)).astype(np.int64).clip(1, n_max_pairs)

    annotation_ng_df = annotation_df[annotation_df['original_y'] != 0]
    ng_material_defect_count_df = annotation_ng_df.groupby(['material_id', 'y']).count()[['original_y']]
    ng_material_id_counts = ng_material_defect_count_df.to_numpy().flatten()
    # ng_material_id_list, ng_material_id_counts = np.unique(annotation_df[annotation_df['original_y'] != 0]['material_id'], return_counts=True)
    k_sample_list = (np.max(ng_material_id_counts)*(1 / ng_material_id_counts)).astype(np.int64).clip(1, n_max_pairs)
    ng_material_defect_count_df['count'] = k_sample_list
    ng_material_defect_count_dict = ng_material_defect_count_df['count'].to_dict()
    # start sampling ok ng pairs
    Xpairs_resampled = []
    ypair_resampled = []
    yrawpair_resampled = []
    material_resampled = []

    sOK_sampled = []
    ok_only_material_id_list = []
    ng_counter = 0
    for material_id in material_id_list:

        material_id_df = annotation_df[annotation_df['material_id'] == material_id].copy()
        material_sNG_df = material_id_df[material_id_df['original_y'] != 0]
        material_sOK_df = material_id_df[material_id_df['original_y'] == 0]

        if len(material_sNG_df) == 0:
            ok_only_material_id_list.append(material_id)
            continue

        for i, sNG_info in material_sNG_df.iterrows():
            sNG_cad = sNG_info['cad']
            sNG_side = sNG_info['side']
            sNG_angle = sNG_info['p_angle']
            y = sNG_info['y']
            k_sample = ng_material_defect_count_dict[(material_id, y)]

            # add in pairs with defective instance and its corresponding ok instance
            sOK_same_component_type = material_sOK_df[(material_sOK_df['cad'] == sNG_cad) &
                                                     (material_sOK_df['side'] == sNG_side) &
                                                     (material_sOK_df['p_angle'] == sNG_angle)].copy()
            n_sOK_available = len(sOK_same_component_type)
            if n_sOK_available < k_sample:
                if n_sOK_available == 0:
                    continue
                chosen_sOKs = sOK_same_component_type.sample(n=int(k_sample), replace=True)
            else:
                chosen_sOKs = sOK_same_component_type.sample(n=int(k_sample), replace=False)
            sOK_X_file_paths = list(chosen_sOKs['X_file_path'])
            sOK_y = list(chosen_sOKs['y'])
            sOK_yraw = list(chosen_sOKs['original_y'])

            Xpairs_OK_NG = [[sp, sNG_info['X_file_path']] for sp in sOK_X_file_paths]
            ypair_OK_NG = [[sp, sNG_info['y']] for sp in sOK_y]
            yrawpair_OK_NG = [[sp, sNG_info['original_y']] for sp in sOK_yraw]
            material_OK_NG = list(chosen_sOKs['material_id'])
            ng_counter += len(material_OK_NG)

            # add in pairs with 1 ok instance pair at each side
            sOK_same_component_type_remaining = material_sOK_df[(material_sOK_df['cad'] == sNG_cad) &
                                                     (material_sOK_df['side'] == sNG_side)].copy()
            Xpairs_OK_OK = []
            ypair_OK_OK = []
            yrawpair_OK_OK = []
            material_OK_OK = []
            idx_OK_OK = []

            for angle in [0, -90, 90, 180]:
                sOK_same_component_type_angle_df = sOK_same_component_type_remaining[sOK_same_component_type_remaining['p_angle'] == angle]
                if len(sOK_same_component_type_angle_df) < 1:
                    continue
                elif len(sOK_same_component_type_angle_df) == 1:
                    sOK_pairs_angle_i = sOK_same_component_type_angle_df.sample(n=2, replace=True)
                    Xpairs_OK_OK.append(list(sOK_pairs_angle_i['X_file_path']))
                    ypair_OK_OK.append(list(sOK_pairs_angle_i['y']))
                    yrawpair_OK_OK.append(list(sOK_pairs_angle_i['original_y']))
                    material_OK_OK.append(material_id)
                    idx_OK_OK += list(sOK_pairs_angle_i.index)
                else:
                    ok_k_sample = np.min([int(k_sample//4*2), int(len(sOK_same_component_type_angle_df)*2)])
                    sOK_pair_chosen = sOK_same_component_type_angle_df.sample(n=ok_k_sample, replace=True)
                    sOK_pair_1_indices = list(range(0, ok_k_sample, 2))
                    sOK_pair_2_indices = list(range(1, ok_k_sample + 1, 2))
                    Xpairs_OK_OK += [
                        [list(sOK_pair_chosen['X_file_path'])[i], list(sOK_pair_chosen['X_file_path'])[j]] for i, j
                        in zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                    ypair_OK_OK += [[list(sOK_pair_chosen['y'])[i], list(sOK_pair_chosen['y'])[j]] for i, j in
                                    zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                    yrawpair_OK_OK += [
                        [list(sOK_pair_chosen['original_y'])[i], list(sOK_pair_chosen['original_y'])[j]] for i, j in
                        zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                    material_OK_OK += [material_id for i, j in zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                    idx_OK_OK += list(sOK_pair_chosen.index)

            Xpairs_resampled += Xpairs_OK_NG + Xpairs_OK_OK
            ypair_resampled += ypair_OK_NG + ypair_OK_OK
            yrawpair_resampled += yrawpair_OK_NG + yrawpair_OK_OK
            material_resampled += material_OK_NG + material_OK_OK
            sOK_sampled += idx_OK_OK

    n_defect_pairs = len(material_resampled)

    # for ok sample in each material id, form a self pair
    ok_annotation_df_unused = annotation_df[annotation_df['original_y'] == 0].drop(index=sOK_sampled).copy()

    sOK_self_1_unique_df = ok_annotation_df_unused.groupby(['material_id', 'cad', 'p_angle'], group_keys=False).apply(
        lambda x: x.sample(1, random_state=seed))

    Xpairs_OK_OK_new = []
    ypair_OK_OK_new = []
    yrawpair_OK_OK_new = []
    material_OK_OK_new = []

    n_new_budget = ((k * n_defect_pairs)//batch_size+1)*batch_size - n_defect_pairs
    while len(material_OK_OK_new) < int(n_new_budget):
        # sample a material subset
        material_chosen = random.choices(ok_only_material_id_list, k=1)[0]
        material_id_df = ok_annotation_df_unused[
            ok_annotation_df_unused['material_id'] == material_chosen].copy()
        if len(material_id_df) == 0 :
            continue

        cad_chosen = random.choices(list(material_id_df['cad'].unique()), k=1)[0]
        p_angle_chosen = random.choices(list(material_id_df['p_angle'].unique()), k=1)[0]

        sOK_cad_p_angle_info = material_id_df[
            (material_id_df['cad'] == cad_chosen) & (material_id_df['p_angle'] == p_angle_chosen)]

        if len(sOK_cad_p_angle_info) == 0:
            continue

        sOK_pair_chosen = sOK_cad_p_angle_info.sample(n=2, replace=True)
        Xpairs_OK_OK_new.append(list(sOK_pair_chosen['X_file_path']))
        ypair_OK_OK_new.append(list(sOK_pair_chosen['y']))
        yrawpair_OK_OK_new.append(list(sOK_pair_chosen['original_y']))
        material_OK_OK_new.append(material_chosen)

    Xpairs_resampled += Xpairs_OK_OK_new
    ypair_resampled += ypair_OK_OK_new
    yrawpair_resampled += yrawpair_OK_OK_new
    material_resampled += material_OK_OK_new
    ybinary_resampled = [yrawpair[0] != yrawpair[1] for yrawpair in yrawpair_resampled]
    print(f'train: n_defect_pairs={np.sum(ybinary_resampled)}, n_ok_pairs={len(ybinary_resampled)-np.sum(ybinary_resampled)}, n_total_pairs={len(ybinary_resampled)}')
    return Xpairs_resampled, ypair_resampled, ybinary_resampled, material_resampled


def stratified_train_val_split(annotation_filename, val_ratio=0.3, groupby='defect_label', label_confidence = 'all',
                               region=None, defect_code=None, seed=42, verbose=False):
    alltype_annotation_df = pd.read_csv(annotation_filename, index_col=0)
    if label_confidence == 'certain':
        alltype_annotation_df_certain = alltype_annotation_df[(alltype_annotation_df['confidence'] == 'certain') | (alltype_annotation_df['confidence'] == 'unchecked')].copy()
        print(
            f'train pair certain = {len(alltype_annotation_df_certain)} out of {len(alltype_annotation_df)}')
        alltype_annotation_df = alltype_annotation_df_certain
    # only select the relevant defect class data: defect_code_link: key=old_y, value=new_y
    defect_class_considered = list(defect_code.keys())
    alltype_annotation_df_sub = alltype_annotation_df[[y in defect_class_considered for
                                                         y in alltype_annotation_df['y']]].copy()

    # reset index
    alltype_annotation_df_sub = alltype_annotation_df_sub.reset_index(drop=True)
    alltype_annotation_df_sub['original_y'] = alltype_annotation_df_sub['y']

    # replace wrong with ok
    updated_y = []
    for y in alltype_annotation_df_sub['original_y']:
        updated_y.append(defect_code[y])
    alltype_annotation_df_sub['y'] = updated_y
    alltype_annotation_df_sub['X_file_path'] = alltype_annotation_df_sub['image_path']

    if verbose:
        print('all non paired data histograms')
        print(alltype_annotation_df_sub.groupby(groupby).count())

    if val_ratio == 0:
        train_annotation_df = alltype_annotation_df_sub.copy()
        val_annotation_df = None
    else:
        subgroupby = 'material_id'
        val_annotation_list = []
        for defect_i in alltype_annotation_df_sub[f'defect_label'].unique():
            defect_i_class_df = alltype_annotation_df_sub[alltype_annotation_df_sub[f'defect_label']==defect_i].copy()
            defect_i_val_annotation_df = defect_i_class_df.groupby(subgroupby, group_keys=False).apply(
                lambda x: x.sample(frac=val_ratio, random_state=seed))
            val_annotation_list.append(defect_i_val_annotation_df)

        val_annotation_df = pd.concat(val_annotation_list)
        val_indices = list(val_annotation_df.index)
        train_indices = [i for i in alltype_annotation_df_sub.index if i not in val_indices]
        train_annotation_df = alltype_annotation_df_sub.iloc[train_indices]

        # if verbose:
        #     print('val raw data histograms')
        #     print(val_annotation_df.groupby(groupby).count())

    return train_annotation_df, val_annotation_df

def generate_test_pairs(test_annotation_df, n_max_pairs=50, reference_annotation_df=None):

    material_id_list = list(test_annotation_df['material_id'].unique())

    # ng_material_id_list, ng_material_id_counts = np.unique(
    #     test_annotation_df[test_annotation_df['original_y'] != 0]['material_id'], return_counts=True)

    annotation_ng_df = test_annotation_df[test_annotation_df['original_y'] != 0]
    ng_material_defect_count_df = annotation_ng_df.groupby(['material_id', 'y']).count()[['original_y']]
    ng_material_id_counts = ng_material_defect_count_df.to_numpy().flatten()

    if reference_annotation_df is not None:
        ref_OK_annotation_df = reference_annotation_df[reference_annotation_df['original_y'] == 0]
        k_sample_list = (2*np.max(ng_material_id_counts) * (1 / ng_material_id_counts)).astype(np.int64).clip(1,
                                                                                                          n_max_pairs)
    else:
        k_sample_list = (np.max(ng_material_id_counts) * (1 / ng_material_id_counts)).astype(np.int64).clip(1, n_max_pairs)

    # k_sample_list = (np.max(ng_material_id_counts)*(1 / ng_material_id_counts)).astype(np.int64).clip(1, n_max_pairs)
    ng_material_defect_count_df['count'] = k_sample_list
    ng_material_defect_count_dict = ng_material_defect_count_df['count'].to_dict()

    Xpairs_resampled, ypair_resampled, yrawpair_resampled, material_resampled, sOK_sampled = [], [], [], [], []

    for material_id in material_id_list:

        test_material_id_df = test_annotation_df[test_annotation_df['material_id'] == material_id].copy()
        test_material_sNG_df = test_material_id_df[test_material_id_df['original_y'] != 0]
        test_material_sOK_df = test_material_id_df[test_material_id_df['original_y'] == 0]
        if reference_annotation_df is not None:
            ref_material_OK_annotation_df = ref_OK_annotation_df[
                ref_OK_annotation_df['material_id'] == material_id].copy()

        # form ng pair
        if len(test_material_sNG_df) > 0:

            # k_sample = k_sample_list[np.where(ng_material_id_list == material_id)][0]

            for i, sNG_info in test_material_sNG_df.iterrows():
                sNG_cad = sNG_info['cad']
                sNG_side = sNG_info['side']
                sNG_angle = sNG_info['p_angle']
                y = sNG_info['y']
                k_sample = ng_material_defect_count_dict[(material_id, y)]
                # add in pairs with defective instance and its corresponding ok instance
                test_sOK_same_component_type = test_material_sOK_df[(test_material_sOK_df['cad'] == sNG_cad) &
                                                                  (test_material_sOK_df['side'] == sNG_side) &
                                                                  (test_material_sOK_df['p_angle'] == sNG_angle)].copy()
                n_sOK_available = len(test_sOK_same_component_type)
                if n_sOK_available < k_sample:
                    if reference_annotation_df is not None:
                        ref_sOK_same_component_type = ref_material_OK_annotation_df[
                            (ref_material_OK_annotation_df['cad'] == sNG_cad) &
                            (ref_material_OK_annotation_df['side'] == sNG_side) &
                            (ref_material_OK_annotation_df['p_angle'] == sNG_angle)].copy()
                        sOK_same_component_type = pd.concat([test_sOK_same_component_type, ref_sOK_same_component_type])
                        n_sOK_available = len(sOK_same_component_type)
                        if n_sOK_available < k_sample:
                            if n_sOK_available == 0:
                                continue
                            chosen_sOKs = sOK_same_component_type.sample(n=int(k_sample), replace=True)
                        else:
                            chosen_sOKs = sOK_same_component_type.sample(n=int(k_sample), replace=False)
                    else:

                        if n_sOK_available == 0:
                            continue
                        chosen_sOKs = test_sOK_same_component_type.sample(n=int(k_sample), replace=True)
                else:
                    chosen_sOKs = test_sOK_same_component_type.sample(n=int(k_sample), replace=False)

                sOK_X_file_paths = list(chosen_sOKs['X_file_path'])
                sOK_y = list(chosen_sOKs['y'])
                sOK_yraw = list(chosen_sOKs['original_y'])

                Xpairs_OK_NG = [[sp, sNG_info['X_file_path']] for sp in sOK_X_file_paths]
                ypair_OK_NG = [[sp, sNG_info['y']] for sp in sOK_y]
                yrawpair_OK_NG = [[sp, sNG_info['original_y']] for sp in sOK_yraw]
                material_OK_NG = list(chosen_sOKs['material_id'])
                idx_OK_NG = list(chosen_sOKs.index)

                # path_ng = os.path.join('/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data/', sNG_info['X_file_path'])
                # path_ok =  os.path.join('/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data/', sOK_X_file_paths[0])
                # visualize_img_pair(path_ng, path_ok)

                # add in pairs with 1 ok instance pair at each side
                sOK_same_component_type = test_material_sOK_df[(test_material_sOK_df['cad'] == sNG_cad) &
                                                                    (test_material_sOK_df['side'] == sNG_side)].copy()

                Xpairs_OK_OK = []
                ypair_OK_OK = []
                yrawpair_OK_OK = []
                material_OK_OK = []
                idx_OK_OK = []

                for angle in [0, -90, 90, 180]:
                    sOK_same_component_type_angle_df = sOK_same_component_type[
                        sOK_same_component_type['p_angle'] == angle]
                    if len(sOK_same_component_type_angle_df) < 1:
                        continue
                    elif len(sOK_same_component_type_angle_df) == 1:
                        sOK_pairs_angle_i = sOK_same_component_type_angle_df.sample(n=2, replace=True)
                        Xpairs_OK_OK.append(list(sOK_pairs_angle_i['X_file_path']))
                        ypair_OK_OK.append(list(sOK_pairs_angle_i['y']))
                        yrawpair_OK_OK.append(list(sOK_pairs_angle_i['original_y']))
                        material_OK_OK.append(material_id)
                        idx_OK_OK += list(sOK_pairs_angle_i.index)
                    else:
                        if reference_annotation_df is not None:
                            ok_k_sample = np.min([8, int(len(sOK_same_component_type_angle_df) * 2)])
                            sOK_pair_chosen = sOK_same_component_type_angle_df.sample(n=ok_k_sample, replace=True)

                        else:
                            ok_k_sample = np.min([4, int(len(sOK_same_component_type_angle_df) // 2 * 2)])
                            sOK_pair_chosen = sOK_same_component_type_angle_df.sample(n=ok_k_sample, replace=False)
                        sOK_pair_1_indices = list(range(0, ok_k_sample, 2))
                        sOK_pair_2_indices = list(range(1, ok_k_sample + 1, 2))
                        Xpairs_OK_OK += [
                            [list(sOK_pair_chosen['X_file_path'])[i], list(sOK_pair_chosen['X_file_path'])[j]] for i, j
                            in zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                        ypair_OK_OK += [[list(sOK_pair_chosen['y'])[i], list(sOK_pair_chosen['y'])[j]] for i, j in
                                            zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                        yrawpair_OK_OK += [
                            [list(sOK_pair_chosen['original_y'])[i], list(sOK_pair_chosen['original_y'])[j]] for i, j in
                            zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                        material_OK_OK += [material_id for i, j in zip(sOK_pair_1_indices, sOK_pair_2_indices)]

                Xpairs_resampled += Xpairs_OK_NG + Xpairs_OK_OK
                ypair_resampled += ypair_OK_NG + ypair_OK_OK
                yrawpair_resampled += yrawpair_OK_NG + yrawpair_OK_OK
                material_resampled += material_OK_NG + material_OK_OK
                sOK_sampled += idx_OK_NG + idx_OK_OK

                # Xpairs_resampled += Xpairs_OK_NG
                # ypair_resampled += ypair_OK_NG
                # yrawpair_resampled += yrawpair_OK_NG
                # material_resampled += material_OK_NG
                # sOK_sampled += idx_OK_NG
        else:
            Xpairs_OK_OK_new, ypair_OK_OK_new, yrawpair_OK_OK_new, material_OK_OK_new = [], [], [], []
            if len(test_material_sOK_df) == 0:
                continue
            for j, sOK_info in test_material_sOK_df.iterrows():
                sOK_cad = sOK_info['cad']
                sOK_side = sOK_info['side']
                ok_material_cad_df = test_material_sOK_df[(test_material_sOK_df['cad'] == sOK_cad) &
                                                          (test_material_sOK_df['side'] == sOK_side)].copy()
                if len(ok_material_cad_df) == 0:
                    continue
                for p_angle_chosen in ok_material_cad_df['p_angle'].unique():
                    sOK_cad_p_angle_info = ok_material_cad_df[ok_material_cad_df['p_angle'] == p_angle_chosen]
                    if len(sOK_cad_p_angle_info) == 0:
                        continue
                    elif len(sOK_cad_p_angle_info) == 1:
                        sOK_pairs_angle_i_new = sOK_cad_p_angle_info.sample(n=2, replace=True)
                        Xpairs_OK_OK_new.append(list(sOK_pairs_angle_i_new['X_file_path']))
                        ypair_OK_OK_new.append(list(sOK_pairs_angle_i_new['y']))
                        yrawpair_OK_OK_new.append(list(sOK_pairs_angle_i_new['original_y']))
                        material_OK_OK_new.append(material_id)

                    else:
                        if reference_annotation_df is not None:
                            ok_k_sample = np.min([2, int(len(sOK_cad_p_angle_info)*2)])
                            sOK_pair_chosen = sOK_cad_p_angle_info.sample(n=ok_k_sample, replace=True)
                        else:
                            ok_k_sample = np.min([2, int(len(sOK_cad_p_angle_info)//2*2)])
                            sOK_pair_chosen = sOK_cad_p_angle_info.sample(n=ok_k_sample, replace=False)
                        sOK_pair_1_indices = list(range(0, ok_k_sample, 2))
                        sOK_pair_2_indices = list(range(1, ok_k_sample + 1, 2))
                        Xpairs_OK_OK_new += [[list(sOK_pair_chosen['X_file_path'])[i], list(sOK_pair_chosen['X_file_path'])[j]] for i,j in zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                        ypair_OK_OK_new += [[list(sOK_pair_chosen['y'])[i], list(sOK_pair_chosen['y'])[j]] for i,j in zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                        yrawpair_OK_OK_new += [[list(sOK_pair_chosen['original_y'])[i], list(sOK_pair_chosen['original_y'])[j]] for i,j in zip(sOK_pair_1_indices, sOK_pair_2_indices)]
                        material_OK_OK_new += [material_id for i,j in zip(sOK_pair_1_indices, sOK_pair_2_indices)]

            Xpairs_resampled += Xpairs_OK_OK_new
            ypair_resampled += ypair_OK_OK_new
            yrawpair_resampled += yrawpair_OK_OK_new
            material_resampled += material_OK_OK_new

    ybinary_resampled = [yrawpair[0] != yrawpair[1] for yrawpair in yrawpair_resampled]
    if reference_annotation_df is not None:
        print(f'val: n_defect_pairs={np.sum(ybinary_resampled)}, n_ok_pairs={len(ybinary_resampled) - np.sum(ybinary_resampled)}, n_total_pairs={len(ybinary_resampled)}')
    else:
        print(f'test: n_defect_pairs={np.sum(ybinary_resampled)}, n_ok_pairs={len(ybinary_resampled) - np.sum(ybinary_resampled)}, n_total_pairs={len(ybinary_resampled)}')

    return Xpairs_resampled, ypair_resampled, yrawpair_resampled, ybinary_resampled, material_resampled

if __name__ == '__main__':
    region = 'padgroup'
    # data_cleaning_folder = '/Users/binxinru/Documents/StartUp/Projects/Datasets/TmptCleaning/'
    data_cleaning_folder = '/mnt/c/Shiyuan/data/DL/DefectClassificationDataPlatform/tmptcheck/'
    annotation_folder = os.path.join(data_cleaning_folder, f'merged_annotation', '240412')
    annotation_filename = os.path.join(annotation_folder, f'train_labels_{region}.csv')
    val_annotation_filename = os.path.join(annotation_folder, f'val_labels_{region}.csv')
    seed = 42
    batch_size = 256
    k_factor = 2.5
    n_max_pairs_val = 15
    n_max_pairs_train = 20
    defect_name_map = {b: a for a, b in zip(['DEFECT_TYPE_NONE', 'DEFECT_TYPE_MISSING', 'DEFECT_TYPE_INSUFFICIENT_SOLDER',
                               'DEFECT_TYPE_PSEUDO_SOLDER', 'DEFECT_TYPE_SOLDER_SHORTAGE'], [0, 1,4,6,7])}
    defect_code = {'ok': 0, 'missing': 1, 'undersolder': 4, 'pseudosolder': 6, 'solder_shortage': 7}
    defect_code_val = {'ok': 0, 'missing': 1, 'undersolder': 4, 'pseudosolder': 6, 'solder_shortage': 7}
    defect_code_link = {v: i for i, v in enumerate(defect_code.values())}
    defect_decode = {v: k for k, v in defect_code_link.items()}
    defect_class_considered_val = list(defect_code_val.values())
    label_confidence = 'certain'
    alltrain_annotation_df = pd.read_csv(annotation_filename, index_col=0)
    train_annotation_df, val_annotation_df = stratified_train_val_split(annotation_filename, val_ratio=0,
                                                                        defect_code=defect_code_link, region=region,
                                                                        groupby=f'defect_label', seed=seed,
                                                                        label_confidence=label_confidence,
                                                                        verbose=True)
    Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = mp_weighted_resampler(
        train_annotation_df, n_max_pairs=n_max_pairs_train, batch_size=batch_size, k=k_factor)
    train_pair_data = [x + ytrain_resampled[i] + [ybinary_resampled[i], material_train_resampled[i]]
                     for i, x in enumerate(Xtrain_resampled)]
    train_pair_data_df = pd.DataFrame(train_pair_data,
                                columns=['ref_image', 'insp_image', 'ref_y', 'insp_y','binary_y', 'material_id'])
    train_pair_data_df['lighting_type'] = 'RGB'
    train_pair_data_df['adamtask_id'] = 'padgroupv090'
    train_pair_data_df['datasplit'] = 'train'
    train_pair_data_df['insp_y_raw'] = [defect_decode[y] for y in train_pair_data_df['insp_y']]
    train_pair_data_df['ref_y_raw'] = [defect_decode[y] for y in train_pair_data_df['ref_y']]

    train_pair_data_df['insp_defect_label'] = [defect_name_map[y] for y in train_pair_data_df['insp_y_raw']]
    train_pair_data_df['ref_defect_label'] = [defect_name_map[y] for y in train_pair_data_df['ref_y_raw']]

    cad_hash_list = []
    p_angle_list = []
    confidence_list = []
    side_list = []
    ref_xy_list = []
    insp_xy_list = []
    for i, train_anno in train_pair_data_df.iterrows():
        insp_image = train_anno['insp_image']
        ref_image = train_anno['ref_image']
        ref_info_all = alltrain_annotation_df[alltrain_annotation_df['image_path'] == ref_image]
        ref_info = alltrain_annotation_df.iloc[ref_info_all.index[0]]
        insp_info_all = alltrain_annotation_df[alltrain_annotation_df['image_path'] == insp_image]
        insp_info = alltrain_annotation_df.iloc[insp_info_all.index[0]]

        cad = insp_info['cad']
        side = insp_info['side']
        p_angle = insp_info['p_angle']
        confidence = insp_info['confidence']
        inspp_x ,inspp_y = insp_info['p_x'], insp_info['p_y']
        refp_x ,refp_y = ref_info['p_x'], ref_info['p_y']

        insp_xy = f'{int(inspp_x)}_{int(inspp_y)}'
        ref_xy = f'{int(refp_x)}_{int(refp_y)}'

        cad_hash_list.append(cad)
        p_angle_list.append(p_angle)
        confidence_list.append(confidence)
        ref_xy_list.append(ref_xy)
        insp_xy_list.append(insp_xy)
        side_list.append(side)

    train_pair_data_df['side'] = side_list
    train_pair_data_df['cad_hash'] = cad_hash_list
    train_pair_data_df['p_angle'] = p_angle_list
    train_pair_data_df['confidence'] = confidence_list
    train_pair_data_df['ref_xy'] = ref_xy_list
    train_pair_data_df['insp_xy'] = insp_xy_list
    print(len(train_pair_data_df))
    train_pair_data_df.to_csv(os.path.join(annotation_folder, f'aug_train_pair_labels_{region}nonpaired_rgb.csv'))


    val_annotation_df_raw = pd.read_csv(val_annotation_filename, index_col=0)
    if label_confidence == 'certain':
        val_annotation_df_certain = val_annotation_df_raw[
            (val_annotation_df_raw['confidence'] == 'certain') | (
                    val_annotation_df_raw['confidence'] == 'unchecked')].copy()
        print(f'aug val pair certain = {len(val_annotation_df_certain)} out of {len(val_annotation_df_raw)}')
        val_annotation_df_raw = val_annotation_df_certain
    val_annotation_df_raw['original_y'] = val_annotation_df_raw['y']

    val_annotation_df_new = val_annotation_df_raw[[y in defect_class_considered_val for y in
                                                   val_annotation_df_raw['original_y']]].copy().reset_index(
        drop=True)
    val_annotation_df_new['y'] = [defect_code_link[y] for y in val_annotation_df_new['original_y']]
    val_annotation_df_new['X_file_path'] = val_annotation_df_new[f'image_path']

    val_annotation_df = val_annotation_df_new.reset_index(drop=True)
    Xval, yval, yvalraw, ybinary_val, material_val = generate_test_pairs(val_annotation_df, n_max_pairs=n_max_pairs_val,
                                                                         reference_annotation_df=train_annotation_df)
    val_pair_data = [x + yval[i] + [ybinary_val[i], material_val[i]]
                       for i, x in enumerate(Xval)]
    val_pair_data_df = pd.DataFrame(val_pair_data,
                                      columns=['ref_image', 'insp_image', 'ref_y', 'insp_y', 'binary_y', 'material_id'])
    val_pair_data_df['lighting_type'] = 'RGB'
    val_pair_data_df['adamtask_id'] = 'padgroupv090'
    val_pair_data_df['datasplit'] = 'val'
    val_pair_data_df['insp_y_raw'] = [defect_decode[y] for y in val_pair_data_df['insp_y']]
    val_pair_data_df['ref_y_raw'] = [defect_decode[y] for y in val_pair_data_df['ref_y']]
    val_pair_data_df['insp_defect_label'] = [defect_name_map[y] for y in val_pair_data_df['insp_y_raw']]
    val_pair_data_df['ref_defect_label'] = [defect_name_map[y] for y in val_pair_data_df['ref_y_raw']]

    cad_hash_list = []
    p_angle_list = []
    confidence_list = []
    side_list = []
    ref_xy_list = []
    insp_xy_list = []
    alltype_annotation_df = pd.concat([alltrain_annotation_df, val_annotation_df_raw]).drop_duplicates().reset_index(drop=True)
    for i, val_anno in val_pair_data_df.iterrows():
        insp_image = val_anno['insp_image']
        ref_image = val_anno['ref_image']
        ref_info_all = alltype_annotation_df[alltype_annotation_df['image_path'] == ref_image]
        ref_info = alltype_annotation_df.iloc[ref_info_all.index[0]]

        insp_info_all = alltype_annotation_df[alltype_annotation_df['image_path'] == insp_image]
        insp_info = alltype_annotation_df.iloc[insp_info_all.index[0]]

        cad = insp_info['cad']
        side = insp_info['side']
        p_angle = insp_info['p_angle']
        confidence = insp_info['confidence']
        inspp_x, inspp_y = insp_info['p_x'], insp_info['p_y']
        refp_x, refp_y = ref_info['p_x'], ref_info['p_y']

        insp_xy = f'{int(inspp_x)}_{int(inspp_y)}'
        ref_xy = f'{int(refp_x)}_{int(refp_y)}'

        cad_hash_list.append(cad)
        p_angle_list.append(p_angle)
        confidence_list.append(confidence)
        ref_xy_list.append(ref_xy)
        insp_xy_list.append(insp_xy)
        side_list.append(side)

    val_pair_data_df['side'] = side_list
    val_pair_data_df['cad_hash'] = cad_hash_list
    val_pair_data_df['p_angle'] = p_angle_list
    val_pair_data_df['confidence'] = confidence_list
    val_pair_data_df['ref_xy'] = ref_xy_list
    val_pair_data_df['insp_xy'] = insp_xy_list
    print(len(val_pair_data_df))
    val_pair_data_df.to_csv(os.path.join(annotation_folder, f'aug_val_pair_labels_{region}nonpaired_rgb.csv'))

    defect_pairs_df_list = [train_pair_data_df, val_pair_data_df]
    for file in ['aug_train_pair_labels_padgroup_merged', 'aug_val_pair_labels_padgroup_merged', 'bbtest_labels_pairs_padgroup_230306v2']:
        pair_df = pd.read_csv(os.path.join(annotation_folder, f'{file}.csv'), index_col=0)
        pair_df_sub = pair_df[[y in defect_class_considered_val for y in pair_df['insp_y_raw']]].copy().reset_index(
            drop=True)
        defect_pairs_df_list.append(pair_df_sub)
    defect_pairs_df = pd.concat(defect_pairs_df_list).drop_duplicates().reset_index(drop=True)
    defect_pairs_df.to_csv(os.path.join(annotation_folder, f'defect_labels_{region}_RGB_pairs_240412.csv'))
    defect_pairs_df['ref_image_name'] = [f.split('/')[-1] for f in defect_pairs_df['ref_image']]
    defect_pairs_df['insp_image_name'] = [f.split('/')[-1] for f in defect_pairs_df['insp_image']]

    all_ref_df = defect_pairs_df[['ref_image_name', 'ref_y_raw']]
    all_insp_df = defect_pairs_df[['insp_image_name', 'insp_y_raw']]
    all_ref_df.columns = ['image_name', 'y_raw']
    all_insp_df.columns = ['image_name', 'y_raw']
    all_img_df = pd.concat([all_ref_df, all_insp_df]).drop_duplicates().reset_index(drop=True)
    all_img_df['defect_label'] = [defect_name_map[y] for y in all_img_df['y_raw']]
    all_img_df.to_csv(os.path.join(annotation_folder, f'defect_image_labels_{region}_rgb.csv'))

