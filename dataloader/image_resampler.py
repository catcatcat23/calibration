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
import cv2
from PIL import Image
from utils.utilities import compress_img


class ImageLoader2(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path_pairs, label_pairs, binary_label_pairs, 
                 positions, compression_p = 0, p_range = [65, 95], transform=None, transform_same=None, transform_sync=None):
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

    def __getitem__(self, index):
        img1_path, img2_path = self.img_path_pairs[index]
        s1_y, s2_y = self.label_pairs[index]  # 缺陷标签
        binary_y = self.binary_label_pairs[index]
        position = self.positions[index]
        s1_img_path = os.path.join(self.root_folder, img1_path)
        s2_img_path = os.path.join(self.root_folder, img2_path)

        try:
            if self.compression_p and random.random() < self.compression_p:
                random.seed(self.seed)
                s1_X_img =  Image.open(s1_img_path)
                s2_X_img =  Image.open(s2_img_path)

                p = random.randint(self.p_range[0], self.p_range[1])
                select_p = random.random()
                s1_X = compress_img(s1_X_img, p, select_p) / 255
                s2_X = compress_img(s2_X_img, p, select_p) / 255
            else:
                s1_X = read_image(os.path.join(self.root_folder, img1_path)) / 255
                s2_X = read_image(os.path.join(self.root_folder, img2_path)) / 255
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

        return s1_X, s2_X, s1_y, s2_y, binary_y, position

    def __len__(self):
        return len(self.label_pairs)
def visual_tensor_img(s1_X):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(s1_X.permute(1, 2, 0))
    plt.title('s1')
    plt.show()
    #
    # plt.figure()
    # plt.imshow(s2_X.permute(1, 2, 0))
    # plt.title('s2')
    # plt.show()

class LUT:
    def __init__(self, lut_path, lut_p):
        self.lut_img = cv2.imread(lut_path, cv2.IMREAD_COLOR)
        self.lut_p = lut_p

    def __call__(self, x):
        if np.random.rand() <= self.lut_p:
            x_np = (x*255).numpy().astype(np.uint8).transpose(1, 2, 0)
            lut_x_np = cv2.LUT(x_np, self.lut_img)
            lut_x_tensor = torch.tensor(lut_x_np.transpose(2, 0, 1)) / 255
        else:
            lut_x_tensor = x
        return lut_x_tensor
    
class DiscreteRotate:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class OneSidePadding:
    # TODO: fix me
    def __init__(self, ratio: float):
        self.ratio = ratio
    def __call__(self, img):

        nchannel, height, width  = img.shape
        if height/width < self.ratio:
            new_height = int(width * self.ratio)
            pad_y = int(new_height - height)
            top, bottom = 0, round(pad_y - 0.1)
            padding = (0, top, 0, bottom)

        elif height/width >= self.ratio:
            new_width = int(height / self.ratio)
            pad_x = int(new_width - width)
            left, right = 0, round(pad_x - 0.1)
            padding = (left, 0, right, 0)

        img_padded = TF.pad(img, padding)

        return img_padded


class AugmentChannel:
    def __init__(self, channel_id: int):
        self.channel_id = channel_id
    def __call__(self, img):

        channel_selected = img.shape
        if width > height:
            new_height = int(width * self.ratio)
            new_width = width
            pad_y = int(new_height - height)
            top, bottom = round(pad_y + 0.1), round(pad_y - 0.1)
            padding = (0, top, 0, bottom)

        else:
            new_height = height
            new_width = int(height * self.ratio)
            pad_x = int(new_width - width)
            left, right = round(pad_x + 0.1), round(pad_x - 0.1)
            padding = (left, 0, right, 0)

        img_padded = TF.pad(img, padding, 255 // 2)

        return img_padded

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
        else:
            self.comp_func = compress_img_opencv

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

class ImageLoader2(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path_pairs, label_pairs, binary_label_pairs, 
                 positions, compression_p = 0, p_range = [65, 95], transform=None, transform_same=None, transform_sync=None):
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
        s1_y, s2_y = self.label_pairs[index]
        binary_y = self.binary_label_pairs[index]
        position = self.positions[index]
 
        # print(img1_path)
        # print(img2_path)
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
        else:
            # print(f"self.root_folder: {self.root_folder}")
            # print(f"img1_path : {img1_path}")
            # print(f"img2_path : {img2_path}")
            # print(f"s1_X : {s1_X}")
            # print(f"s2_X : {s2_X}")
            # 非同步的
            s1_X = self.transform(s1_X)
            s2_X = self.transform(s2_X)

        if self.transform_sync is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s1_X = self.transform_sync(s1_X)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s2_X = self.transform_sync(s2_X)
            self.seed += 1

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(2, 1)
            # axes[0].imshow(s1_X.permute(1, 2, 0), cmap='gray')
            # axes[0].set_title('S1')
            # axes[0].set_xticks([])
            # axes[0].set_yticks([])
            #
            # axes[1].imshow(s2_X.permute(1, 2, 0), cmap='gray')
            # axes[1].set_title(f'S2')
            # axes[1].set_xticks([])
            # axes[1].set_yticks([])
            # plt.tight_layout()
            # plt.show()

        return s1_X, s2_X, s1_y, s2_y, binary_y, position

    def __len__(self):
        return len(self.label_pairs)


class ImageLoaderTriplet(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path_pairs, label_pairs, binary_label_pairs, positions, transform_crop=None, transform=None,
                 transform_perturb = None, transform_same=None, transform_sync=None):
        self.root_folder = root_folder
        self.img_path_pairs = img_path_pairs
        self.label_pairs = label_pairs
        self.binary_label_pairs = binary_label_pairs
        self.positions = positions
        self.transform = transform
        self.transform_sync  = transform_sync
        self.transform_same = transform_same
        self.transform_perturb = transform_perturb
        self.seed = 0
        self.transform_crop = transform_crop

    def __getitem__(self, index):
        img1_path, img2_path = self.img_path_pairs[index]
        s1_y, s2_y = self.label_pairs[index]
        s3_y = s2_y
        binary_y = self.binary_label_pairs[index]
        position = self.positions[index]
        try:
            s1_X_raw = read_image(os.path.join(self.root_folder, img1_path)) / 255
            s2_X_raw = read_image(os.path.join(self.root_folder, img2_path)) / 255
            if s1_X_raw.shape[0] == 4 or s2_X_raw.shape[0] == 4:
                s1_X_raw = s1_X_raw[:3]
                s2_X_raw = s2_X_raw[:3]
        except:
            print(img1_path)
            print(img2_path)
            print(s1_y, s2_y)

        if img1_path == img2_path:

            if self.transform_same is not None:
                s1_X = self.transform_same(s1_X_raw)
                s2_X = self.transform_same(s2_X_raw)
                s3_X = self.transform_same(s2_X_raw)
            else:
                s1_X = self.transform(s1_X_raw)
                s1_X = self.transform_crop(s1_X)
                s2_X = self.transform(s2_X_raw)
                s2_X = self.transform_crop(s2_X)
                s3_X = self.transform(s2_X_raw)
                s3_X = self.transform_crop(s3_X)

        else:
            s1_X = self.transform(s1_X_raw)
            s1_X = self.transform_crop(s1_X)
            s2_X_tmpt = self.transform(s2_X_raw)
            s2_X = self.transform_crop(s2_X_tmpt)
            s3_X = self.transform_perturb(s2_X)

        if self.transform_sync is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s1_X = self.transform_sync(s1_X)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s2_X = self.transform_sync(s2_X)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            s3_X = self.transform_sync(s3_X)

            self.seed += 1

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(2, 1)
            # axes[0].imshow(s1_X.permute(1, 2, 0), cmap='gray')
            # axes[0].set_title('S1')
            # axes[0].set_xticks([])
            # axes[0].set_yticks([])
            #
            # axes[1].imshow(s2_X.permute(1, 2, 0), cmap='gray')
            # axes[1].set_title(f'S2')
            # axes[1].set_xticks([])
            # axes[1].set_yticks([])
            # plt.tight_layout()
            # plt.show()

        return s1_X, s2_X, s3_X, s1_y, s2_y, s3_y, binary_y, position

    def __len__(self):
        return len(self.label_pairs)

class ImageLoaderSingle(torch.utils.data.Dataset):
    def __init__(self, root_folder, img_path, label, transform=None):
        self.root_folder = root_folder
        self.img_paths = img_path
        self.labels = label
        self.transform = transform

    def __getitem__(self, index):
        y = self.labels[index]
        X = read_image(os.path.join(self.root_folder, self.img_paths[index])) / 255
        if self.transform is not None:
            X = self.transform(X)

        return X, y

    def __len__(self):
        return len(self.labels)


def mp_weighted_resampler(annotation_df, k=3, batch_size=64, n_max_pairs=10, p_qualified=0.75, seed=42):
    # np.random.seed(seed)
    # random.seed(seed)

    material_ids = list(annotation_df['material_id'].unique())
    n_total = len(annotation_df)
    # prepare position subset
    material_subsets = {material_i: {
        'X_file_path': list(annotation_df[annotation_df['material_id'] == material_i]['X_file_path']),
        'y': list(annotation_df[annotation_df['material_id'] == material_i]['y']),
        'y_raw': list(annotation_df[annotation_df['material_id'] == material_i]['original_y']),
        'cad': list(annotation_df[annotation_df['material_id'] == material_i]['cad']),
        'side': list(annotation_df[annotation_df['material_id'] == material_i]['side']),
        'counts': len(annotation_df[annotation_df['material_id'] == material_i]['X_file_path']),
        'defect_counts': len(
            annotation_df[(annotation_df['material_id'] == material_i) & (annotation_df['original_y'] != 0)][
                'original_y']),
    } for material_i in material_ids}

    ng_material_id_list, ng_material_id_counts = np.unique(
        annotation_df[annotation_df['original_y'] != 0]['material_id'], return_counts=True)
    p_ng_material_norm = np.array(ng_material_id_counts) / sum(ng_material_id_counts)
    ng_k_sample_list = (np.max(ng_material_id_counts)*(1 / ng_material_id_counts)).astype(np.int64).clip(1, n_max_pairs)

    ok_material_id_list, ok_material_id_counts = np.unique(
        annotation_df[annotation_df['original_y'] == 0]['material_id'], return_counts=True)
    p_ok_material_norm = np.array(ok_material_id_counts) / sum(ok_material_id_counts)

    # start resampling
    Xpairs_resampled = []
    ypair_resampled = []
    yrawpair_resampled = []
    material_resampled = []
    ok_X_path_sampled_for_ng = []
    # add in pairs with defective instance and its corresponding ok instance
    for i, material_id in enumerate(ng_material_id_list):
        ng_k_pairs = ng_k_sample_list[i]
        material_values = material_subsets[material_id]
        material_id_X_file_paths = material_values['X_file_path']
        material_id_yraws = material_values['y_raw']
        material_id_ys = material_values['y']
        material_id_cad = material_values['cad']
        material_id_side = material_values['side']

        s1_X_file_paths_y = [[X_path, material_id_ys[i], material_id_yraws[i], material_id_cad[i], material_id_side[i]]
                             for i, X_path in enumerate(material_id_X_file_paths) if material_id_yraws[i] == 0]
        s2_X_file_paths_y = [[X_path, material_id_ys[i], material_id_yraws[i], material_id_cad[i], material_id_side[i]]
                             for i, X_path in enumerate(material_id_X_file_paths) if material_id_yraws[i] != 0]

        for s2_Xy in s2_X_file_paths_y:
            s2_cad = s2_Xy[3]
            s2_side = s2_Xy[4]
            s1_same_cad_X_file_paths_y = [s1 for s1 in s1_X_file_paths_y if s1[3] == s2_cad]
            s1_same_board_X_file_paths_y = [s1 for s1 in s1_X_file_paths_y if s1[3] == s2_cad and s1[4] == s2_side]
            if len(s1_same_board_X_file_paths_y) == 0:
                if len(s1_same_cad_X_file_paths_y) == 0:
                    continue
                else:
                    s1_same_board_X_file_paths_y = s1_same_cad_X_file_paths_y

            if len(s1_same_board_X_file_paths_y) >= (ng_k_pairs):
                # print(f'N ok samples = {len(s1_same_board_X_file_paths_y)}')
                chosen_indices = np.random.choice(range(len(s1_same_board_X_file_paths_y)), ng_k_pairs, replace=False)
                s1_same_board_X_file_paths_y_selected = [s1_same_board_X_file_paths_y[i] for i in chosen_indices]
                ok_X_path_sampled_for_ng += [xp[0] for xp in s1_same_board_X_file_paths_y]
            else:
                chosen_indices = np.random.choice(range(len(s1_same_board_X_file_paths_y)), ng_k_pairs, replace=True)
                s1_same_board_X_file_paths_y_selected = [s1_same_board_X_file_paths_y[i] for i in chosen_indices]
            # else:
            #     s1_same_board_X_file_paths_y_selected = s1_same_board_X_file_paths_y.copy()

            Xpairs_resampled += [[s1_Xy[0], s2_Xy[0]] for s1_Xy in s1_same_board_X_file_paths_y_selected]
            ypair_resampled += [[s1_Xy[1], s2_Xy[1]] for s1_Xy in s1_same_board_X_file_paths_y_selected]
            yrawpair_resampled += [[s1_Xy[2], s2_Xy[2]] for s1_Xy in s1_same_board_X_file_paths_y_selected]
            material_resampled += [material_id] * len(s1_same_board_X_file_paths_y_selected)

            if len(s1_same_board_X_file_paths_y) >= 2:
                ok_chosen_indices = np.random.choice(range(len(s1_same_board_X_file_paths_y)), 2, replace=False)
                s1_OK_Xy, s2_OK_Xy = s1_same_board_X_file_paths_y[ok_chosen_indices[0]], s1_same_board_X_file_paths_y[ok_chosen_indices[1]]
                Xpairs_resampled += [[s1_OK_Xy[0], s2_OK_Xy[0]]]
                ypair_resampled += [[s1_OK_Xy[1], s2_OK_Xy[1]]]
                yrawpair_resampled += [[s1_OK_Xy[2], s2_OK_Xy[2]]]
                material_resampled += [material_id]

        # prepare defect sampling probability: increase the defect probability of rare defects
        defect_ids = [y for i, y in enumerate(material_id_yraws) if y != 0]
        unique_defect_ids, unique_defect_id_counts = np.unique(defect_ids, return_counts=True)
        material_values['defect_sampling_p'] = {defect_id: np.exp(-defect_id_counts / len(unique_defect_ids))
                                                for defect_id, defect_id_counts in
                                                zip(unique_defect_ids, unique_defect_id_counts)}

    # for ok sample in each material id, form a self pair
    n_self_sample = 2
    unique_material_id_samples = annotation_df[annotation_df['original_y'] == 0].groupby('material_id', group_keys=False).apply(
        lambda x: x.sample(n_self_sample, random_state=seed, replace=True))
    print(f'unique_material_id = {len(unique_material_id_samples)/n_self_sample}')
    Xpairs_self = [[s_X_path, s_X_path] for s_X_path in list(unique_material_id_samples['X_file_path'])]
    ypair_self = [[s_y, s_y] for s_y in list(unique_material_id_samples['y'])]
    yrawpair_self = [[s_y, s_y] for s_y in list(unique_material_id_samples['original_y'])]
    material_self = list(unique_material_id_samples['material_id'])
    n_self_pairs = int(len(material_self))

    # chosen_self_pairs_ids  = np.random.choice(range(len(material_self)), n_self_pairs, replace=False)
    Xpairs_resampled += Xpairs_self
    ypair_resampled += ypair_self
    yrawpair_resampled += yrawpair_self
    material_resampled += material_self

    n_defect_self_pairs = len(material_resampled)

    Xpairs_resampled_new = []
    ypair_resampled_new = []
    yrawpair_resampled_new = []
    material_resampled_new = []
    if n_defect_self_pairs == n_self_pairs:
        remaining_sampling_budget = k*len(annotation_df)
        p_qualified = 1.00
        p_ok_material_norm = np.ones(len(ok_material_id_counts))/len(ok_material_id_counts)
    else:
        remaining_sampling_budget = (int(k * n_defect_self_pairs / batch_size + 1) * batch_size - len(material_resampled))
    while len(material_resampled_new) < remaining_sampling_budget:

        p_rand = np.random.rand()
        if p_rand > p_qualified:
            # sample ok-ng pairs
            material_chosen = random.choices(ng_material_id_list, weights=p_ng_material_norm, k=1)
            X_path_material = material_subsets[material_chosen[0]]['X_file_path']
            y_material = material_subsets[material_chosen[0]]['y']
            yraw_material = material_subsets[material_chosen[0]]['y_raw']
            material_id_cad = material_subsets[material_chosen[0]]['cad']
            material_id_side = material_subsets[material_chosen[0]]['side']
            # check whether material subset contains a qualified sample
            qualified_ids = [i for i, y in enumerate(yraw_material) if
                             y == 0 and X_path_material[i] not in ok_X_path_sampled_for_ng]
            p_defect_ids = [material_subsets[material_chosen[0]]['defect_sampling_p'][y] for i, y in
                            enumerate(yraw_material) if y != 0]
            defect_ids = [i for i, y in enumerate(yraw_material) if y != 0]
            p_defect_ids_norm = p_defect_ids / np.sum(p_defect_ids)
            # randomly sample a defective sample with probability 1-p_qualified
            # upsampled rare defects
            s2_id = random.choices(defect_ids, weights=p_defect_ids_norm, k=1)[0]
        else:
            material_chosen = random.choices(ok_material_id_list, weights=p_ok_material_norm, k=1)
            X_path_material = material_subsets[material_chosen[0]]['X_file_path']
            y_material = material_subsets[material_chosen[0]]['y']
            yraw_material = material_subsets[material_chosen[0]]['y_raw']
            material_id_cad = material_subsets[material_chosen[0]]['cad']
            material_id_side = material_subsets[material_chosen[0]]['side']
            # check whether material subset contains a qualified sample
            qualified_ids = [i for i, y in enumerate(yraw_material) if y == 0]
            if len(qualified_ids) < 2:
                continue
            # randomly sample another qualified sample with probability p_qualified
            s2_id = random.choice(qualified_ids)
            qualified_ids.remove(s2_id)

        s2_X_path = X_path_material[s2_id]
        s2_y = y_material[s2_id]
        s2_yraw = yraw_material[s2_id]
        s2_cad = material_id_cad[s2_id]
        s2_side = material_id_side[s2_id]

        # sample a ok sample
        # remove the previous qualified sample from qualified_ids
        qualified_ids_rest = [i for i in qualified_ids if
                              material_id_cad[i] == s2_cad and material_id_side[i] == s2_side]
        # check whether material subset still contains another qualified sample
        if len(qualified_ids_rest) < 1:
            continue

        # randomly sample a qualified sample from material subset
        s1_id = random.choice(qualified_ids_rest)
        s1_X_path = X_path_material[s1_id]
        s1_y = y_material[s1_id]
        s1_yraw = yraw_material[s1_id]

        # add s1 and s2 to the resampled dataset but prevent redundancy
        # note previous version would prevent resample of all defect pairs formed
        # this version we permit one repeated sample for some defect pairs
        if ([s1_X_path, s2_X_path] not in Xpairs_resampled_new) and (
                [s2_X_path, s1_X_path] not in Xpairs_resampled_new):
            Xpairs_resampled_new.append([s1_X_path, s2_X_path])
            ypair_resampled_new.append([s1_y, s2_y])
            yrawpair_resampled_new.append([s1_yraw, s2_yraw])
            material_resampled_new.append(material_chosen[0])

    Xpairs_resampled += Xpairs_resampled_new
    ypair_resampled += ypair_resampled_new
    yrawpair_resampled += yrawpair_resampled_new
    material_resampled += material_resampled_new
    ybinary_resampled = [yrawpair[0] != yrawpair[1] for yrawpair in yrawpair_resampled]

    return Xpairs_resampled, ypair_resampled, ybinary_resampled, material_resampled

# def stratified_train_val_split(annotation_filename, val_ratio=0.3, groupby='defect_label',
#                                region=None, defect_code=None, seed=42, verbose=False):
#     alltype_annotation_df = pd.read_csv(annotation_filename, index_col=0)
#     # only select the relevant defect class data: defect_code_link: key=old_y, value=new_y
#     defect_class_considered = list(defect_code.keys()) + [3]
#     # defect_class_considered_sub = alltype_annotation_df.copy()
#     # defect_code_aug = defect_code.copy()
#     # for y in defect_class_considered_sub['y']:
#     #     if y not in defect_code_aug:
#     #         defect_code_aug[y] = len(defect_code_aug)
#     defect_class_considered_sub = alltype_annotation_df[[y in defect_class_considered for
#                                                          y in alltype_annotation_df['y']]].copy()
#
#     # reset index
#     defect_class_considered_sub = defect_class_considered_sub.reset_index(drop=True)
#     defect_class_considered_sub['original_y'] = defect_class_considered_sub['y']
#     # defect_class_considered_sub['y'] = [defect_code[y] for y in defect_class_considered_sub['original_y']]
#     # replace wrong with ok
#     updated_y = []
#     for y in defect_class_considered_sub['original_y']:
#         if y == 3:
#             updated_y.append(0)
#         else:
#             updated_y.append(defect_code[y])
#     defect_class_considered_sub['y'] = updated_y
#     defect_class_considered_sub['X_file_path'] = defect_class_considered_sub[f'{region}_file_path_short']
#
#     if verbose:
#         print('all data histograms')
#         print(defect_class_considered_sub.groupby(groupby).count())
#
#     if val_ratio == 0:
#         train_annotation_df = defect_class_considered_sub.copy()
#         val_annotation_df = None
#     else:
#         val_annotation_df = defect_class_considered_sub.groupby(groupby, group_keys=False).apply(
#             lambda x: x.sample(frac=val_ratio, random_state=seed))
#         val_indices = list(val_annotation_df.index)
#         train_indices = [i for i in defect_class_considered_sub.index if i not in val_indices]
#         train_annotation_df = defect_class_considered_sub.iloc[train_indices]
#         if verbose:
#             print('val data histograms')
#             print(val_annotation_df.groupby(groupby).count())
#
#     return train_annotation_df, val_annotation_df

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
    defect_class_considered_sub = alltype_annotation_df[[y in defect_class_considered for
                                                         y in alltype_annotation_df['y']]].copy()

    # reset index
    defect_class_considered_sub = defect_class_considered_sub.reset_index(drop=True)
    defect_class_considered_sub['original_y'] = defect_class_considered_sub['y']
    # defect_class_considered_sub['y'] = [defect_code[y] for y in defect_class_considered_sub['original_y']]
    updated_y = []
    for y in defect_class_considered_sub['original_y']:
        updated_y.append(defect_code[y])
    defect_class_considered_sub['y'] = updated_y
    # replace wrong with ok
    defect_class_considered_sub['X_file_path'] = defect_class_considered_sub[f'image_path']

    if verbose:
        print('all data histograms')
        print(defect_class_considered_sub.groupby(groupby).count())

    if val_ratio == 0:
        train_annotation_df = defect_class_considered_sub.copy()
        val_annotation_df = None
    else:
        val_annotation_df = defect_class_considered_sub.groupby(groupby, group_keys=False).apply(
            lambda x: x.sample(frac=val_ratio, random_state=seed))
        val_indices = list(val_annotation_df.index)
        train_indices = [i for i in defect_class_considered_sub.index if i not in val_indices]
        train_annotation_df = defect_class_considered_sub.iloc[train_indices]
        if verbose:
            print('val data histograms')
            print(val_annotation_df.groupby(groupby).count())

    return train_annotation_df, val_annotation_df


def generate_val_pairs(reference_annotation_df, val_annotation_df):
    ok_reference_annotation_df = pd.concat([reference_annotation_df[(reference_annotation_df['original_y'] == 0)],
                                            val_annotation_df[(val_annotation_df['original_y'] == 0)]])

    Xpairs_resampled, ypair_resampled, yrawpair_resampled, material_resampled = [], [], [], []
    for j in val_annotation_df.index:
        s2_X_path = val_annotation_df.loc[j, 'X_file_path']
        material_chosen = val_annotation_df.loc[j, 'material_id']
        s2_y = val_annotation_df.loc[j, 'y']
        s2_yraw = val_annotation_df.loc[j, 'original_y']

        if material_chosen in list(ok_reference_annotation_df['material_id']):
            ok_reference_material_j = ok_reference_annotation_df[
                ok_reference_annotation_df['material_id'] == material_chosen].copy()
            if s2_yraw == 0:
                ok_reference_material_j = ok_reference_material_j.sample(random_state=j)

            for k, ok_sample_material_j in enumerate(
                    ok_reference_material_j[['X_file_path', 'y', 'original_y']].values.tolist()):
                s1_X_path = ok_sample_material_j[0]
                s1_y = ok_sample_material_j[1]
                s1_yraw = ok_sample_material_j[2]
                Xpairs_resampled.append([s1_X_path, s2_X_path])
                ypair_resampled.append([s1_y, s2_y])
                yrawpair_resampled.append([s1_yraw, s2_yraw])
                material_resampled.append(material_chosen)

    ybinary_resampled = [yrawpair[0] != yrawpair[1] for yrawpair in yrawpair_resampled]

    return Xpairs_resampled, ypair_resampled, yrawpair_resampled, ybinary_resampled, material_resampled


def generate_test_pairs(test_annotation_filename, region=None, defect_code=None, k_ok=4, mode='finetune', replace = True,
                        seed=42, label_confidence = 'all', output_type='dual2'):
    print(f"try generate_test_pairs")
    np.random.seed(seed)
    random.seed(seed)

    # only select the relevant defect class data: defect_code_link: key=old_y, value=new_y
    alltype_annotation_df = pd.read_csv(test_annotation_filename, index_col=0)
    if label_confidence == 'certain':
        alltype_annotation_df_certain = alltype_annotation_df[(alltype_annotation_df['confidence'] == 'certain') | (alltype_annotation_df['confidence'] == 'unchecked')].copy()
        print(
            f'val/test pair certain = {len(alltype_annotation_df_certain)} out of {len(alltype_annotation_df)}')

        alltype_annotation_df = alltype_annotation_df_certain
    # test_annotation_df_full = alltype_annotation_df

    if output_type == 'dual':
        defect_class_considered = list(defect_code.keys()) + [3]
        test_annotation_df_full = alltype_annotation_df[[y in defect_class_considered for
                                                         y in alltype_annotation_df['y']]].copy()
        test_annotation_df_full = test_annotation_df_full.reset_index(drop=True)
        print(f'n_test={len(test_annotation_df_full)}')
        test_annotation_df_full['original_y'] = test_annotation_df_full['y']
        test_annotation_df_full['y'] = [defect_code[y] if y != 3 else 0 for y in test_annotation_df_full['original_y']]
    else:
        defect_class_considered = list(defect_code.keys())
        test_annotation_df_full = alltype_annotation_df[[y in defect_class_considered for
                                                         y in alltype_annotation_df['y']]].copy()
        test_annotation_df_full = test_annotation_df_full.reset_index(drop=True)
        # print(f'n_test={len(test_annotation_df_full)}')
        test_annotation_df_full['original_y'] = test_annotation_df_full['y']
        test_annotation_df_full['y'] = [defect_code[y] for y in test_annotation_df_full['original_y']]

    test_annotation_df_full['X_file_path'] = test_annotation_df_full[f'image_path']

    if mode == 'strict_ref_insp_pair':
        test_annotation_df_full['material_id_raw'] = test_annotation_df_full['material_id'].copy()
        test_annotation_df_full['material_id'] = [f'{p}_{m}' for p, m in zip(test_annotation_df_full[
                                                                                        'position_id'],
                                                                                    test_annotation_df_full[
                                                                                        'material_id_raw'])]
        ok_reference_annotation_df = test_annotation_df_full[
            ['_ref_' in x for x in test_annotation_df_full['X_file_path']]].copy().reset_index(drop=True)
        test_annotation_df = test_annotation_df_full[
            ['_insp_' in x for x in test_annotation_df_full['X_file_path']]].copy().reset_index(drop=True)
    elif mode == 'finetune':
        ok_reference_annotation_df = test_annotation_df_full[
            ['_ref_' in x for x in test_annotation_df_full['X_file_path']]].copy().reset_index(drop=True)
        ok_annotation_df = test_annotation_df_full[(test_annotation_df_full['original_y'] == 0)].copy().reset_index(drop=True)
        test_annotation_df = test_annotation_df_full[['_insp_' in x for x in test_annotation_df_full['X_file_path']]].copy().reset_index(drop=True)
    elif mode == 'ng_only':
        test_annotation_df = test_annotation_df_full
        ok_reference_annotation_df = test_annotation_df[(test_annotation_df['original_y'] == 0)].copy()
        test_annotation_df = test_annotation_df_full[test_annotation_df_full['y']!=0]
    else:
        test_annotation_df = test_annotation_df_full
        ok_reference_annotation_df = test_annotation_df[(test_annotation_df['original_y'] == 0)].copy()
        test_annotation_df = test_annotation_df_full

    Xpairs_resampled, ypair_resampled, yrawpair_resampled, material_resampled = [], [], [], []
    for j in test_annotation_df.index:
        s2_X_path = test_annotation_df.loc[j, 'X_file_path']
        material_chosen = test_annotation_df.loc[j, 'material_id']
        s2_y = test_annotation_df.loc[j, 'y']
        s2_yraw = test_annotation_df.loc[j, 'original_y']
        cad = test_annotation_df.loc[j,'cad']
        side = test_annotation_df.loc[j,'side']

        if mode == 'finetune':
            position_id = test_annotation_df.loc[j, 'position_id']
            ok_reference_material_j = ok_reference_annotation_df[
                (ok_reference_annotation_df['material_id'] == material_chosen) & (
                        ok_reference_annotation_df['cad'] == cad) & (ok_reference_annotation_df['side'] == side)
                & (ok_reference_annotation_df['position_id'] == position_id)].copy()
            ok_material_j = ok_annotation_df[
                (ok_annotation_df['material_id'] == material_chosen) & (
                        ok_annotation_df['cad'] == cad) & (ok_annotation_df['side'] == side)].copy()

        else:
            ok_reference_material_j = ok_reference_annotation_df[(ok_reference_annotation_df['material_id'] == material_chosen) & (
                    ok_reference_annotation_df['cad'] == cad) & (ok_reference_annotation_df['side'] == side)].copy()
        if len(ok_reference_material_j) > 0:
            k_ok_used = np.min([len(ok_reference_material_j), k_ok])
            if s2_yraw == 0:
                ok_reference_material_j = ok_reference_material_j.sample(k_ok_used, random_state=j, replace=replace)
            else:
                if mode == 'finetune':
                    k_ng_used = np.min([np.max([int(len(ok_material_j)/2), k_ok]), 200])

                    print(k_ng_used)
                    ok_reference_material_j = ok_material_j.sample(k_ng_used, random_state=j, replace=False)
                elif mode == 'flexible':
                    ok_reference_material_j = ok_reference_material_j.sample(k_ok_used*4, random_state=j, replace=replace)
                else:
                    ok_reference_material_j = ok_reference_material_j.sample(k_ok_used, random_state=j, replace=replace)

            for k, ok_sample_material_j in enumerate(
                    ok_reference_material_j[['X_file_path', 'y', 'original_y']].values.tolist()):
                s1_X_path = ok_sample_material_j[0]
                if s1_X_path == s2_X_path:
                    continue
                s1_y = ok_sample_material_j[1]
                s1_yraw = ok_sample_material_j[2]

                Xpairs_resampled.append([s1_X_path, s2_X_path])
                ypair_resampled.append([s1_y, s2_y])
                yrawpair_resampled.append([s1_yraw, s2_yraw])
                material_resampled.append(material_chosen)

    ybinary_resampled = [yrawpair[0] != yrawpair[1] for yrawpair in yrawpair_resampled]
    test_data_csv = [x + ypair_resampled[i] + yrawpair_resampled[i] + [ybinary_resampled[i], material_resampled[i]] for
                     i, x in enumerate(Xpairs_resampled)]
    test_data_df = pd.DataFrame(test_data_csv,
                                columns=['ref_image', 'insp_image', 'ref_y', 'insp_y', 'ref_y_raw', 'insp_y_raw',
                                         'binary_y', 'material_id'])

    return Xpairs_resampled, ypair_resampled, yrawpair_resampled, ybinary_resampled, material_resampled, test_data_df


def generate_test_selfpairs(test_annotation_filename, region=None, defect_code=None, k_ok=4, mode='finetune', replace = True, seed=42):

    # only select the relevant defect class data: defect_code_link: key=old_y, value=new_y
    defect_class_considered = list(defect_code.keys()) + [3]
    alltype_annotation_df = pd.read_csv(test_annotation_filename, index_col=0)
    # test_annotation_df_full = alltype_annotation_df
    test_annotation_df_full = alltype_annotation_df[[y in defect_class_considered for
                                                         y in alltype_annotation_df['y']]].copy()
    test_annotation_df_full = test_annotation_df_full.reset_index(drop=True)
    print(f'n_test={len(test_annotation_df_full)}')
    test_annotation_df_full['original_y'] = test_annotation_df_full['y']
    test_annotation_df_full['y'] = [defect_code[y] if y != 3 else 0 for y in test_annotation_df_full['original_y']]
    test_annotation_df_full['X_file_path'] = test_annotation_df_full[f'image_path']

    test_annotation_df = test_annotation_df_full[(test_annotation_df_full['original_y'] == 0)].copy()

    Xpairs_resampled, ypair_resampled, yrawpair_resampled, material_resampled = [], [], [], []
    for j in test_annotation_df.index:
        s2_X_path = test_annotation_df.loc[j, 'X_file_path']
        material_chosen = test_annotation_df.loc[j, 'material_id']
        s2_y = test_annotation_df.loc[j, 'y']
        s2_yraw = test_annotation_df.loc[j, 'original_y']
        cad = test_annotation_df.loc[j,'cad']
        side = test_annotation_df.loc[j,'side']
        # position_id = test_annotation_df.loc[j,'position_id']

        Xpairs_resampled.append([s2_X_path, s2_X_path])
        ypair_resampled.append([s2_y, s2_y])
        yrawpair_resampled.append([s2_yraw, s2_yraw])
        material_resampled.append(material_chosen)

    ybinary_resampled = [yrawpair[0] != yrawpair[1] for yrawpair in yrawpair_resampled]
    test_data_csv = [x + ypair_resampled[i] + yrawpair_resampled[i] + [ybinary_resampled[i], material_resampled[i]] for
                     i, x in enumerate(Xpairs_resampled)]
    test_data_df = pd.DataFrame(test_data_csv,
                                columns=['ref_image', 'insp_image', 'ref_y', 'insp_y', 'ref_yraw', 'insp_yraw',
                                         'binary_y', 'material_id'])

    return Xpairs_resampled, ypair_resampled, yrawpair_resampled, ybinary_resampled, material_resampled, test_data_df


# generate sync data
# n_total = 100
# material_ids = ['C1', 'R1', 'U1','D1']
# X_total = [np.random.random_integers(0,255, (32, 32, 3)) for _ in range(n_total)]
# y_total = np.random.random_integers(0,5, n_total)  # 0 denote qualified sample
# positions_total = random.choices(material_ids, k=n_total)
# # prepare position subset
# position_subsets = {material_i: {'X':[], 'y':[], 'counts': 0} for material_i in material_ids}
# for i, material_i in enumerate(positions_total):
#     position_subsets[material_i]['X'].append(X_total[i])
#     position_subsets[material_i]['y'].append(y_total[i])
#     position_subsets[material_i]['counts'] += 1

# load check data
# def ms_weighted_resampler(annotation_file=None, n_total=None):
# material_ids = ['capacitor','resistor']
if __name__ == '__main__':
    img_folder = '/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data'
    # region_list = ['body','component','pad','pins']
    region_list = ['pad']
    operation = 'bbbox_test'
    if operation == 'real_test':
        # date = '230520A'
        date = 'debug'
        output_type = 'dual2'
        for region in ['component', 'package', 'pad', 'pins']:
            # test_annotation_filename = os.path.join(img_folder, 'merged_annotation', f'real_test',
            #                                         f'real_test_labels_{region}_{date}.csv')
            # test_annotation_filename = os.path.join(img_folder, 'merged_annotation', f'real_test',
            #                                         f'real_test_labels_{region}_{date}.csv')
            test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date,
                                                    f'debug_defect_labels_{region}_rgb.csv')
            if region == 'component':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4,
                                   'oversolder': 5, 'pseudosolder': 6, 'solder_shortage': 7,
                                   'tombstone': 8, 'others': 11}  # v1.31, v1.32

                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'wrong': 3, 'undersolder': 4,
                                   'oversolder': 5, 'pseudosolder': 6, 'solder_shortage': 7,
                                   'tombstone': 8, 'others': 11}  # v1.31, v1.32
            elif region == 'package' or region == 'body':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'tombstone': 8, 'others': 11}  # v1.32
                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}  # v1.32
            elif region == 'pad':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6}  # v1.32
                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6}
            elif region == 'pins':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6,
                                   'solder_shortage': 7}

                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6,
                                   'solder_shortage': 7}

            n_class = len(defect_code)
            defect_code_link = {v: i for i, v in enumerate(defect_code.values())}

            Xtest, ytest, ytestraw, ybinary_test, material_test, test_annotation_df = generate_test_pairs(
                test_annotation_filename, defect_code=defect_code_link, region=region, mode='flexible', k_ok=2, replace=False)
            print(len(test_annotation_df))
            test_annotation_df = test_annotation_df.reset_index(drop=True)
            test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', date,
                                                   f'debug_test_labels_pairs_{region}.csv'))
            # test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', f'real_test',
            #                                        f'real_test_labels_pairs_{region}_{date}.csv'))

    elif operation == 'bbbox_test':
        # date = '230505v2'
        date = '230924'
        for region in region_list:
            test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date,
                                                    f'blackbox_test_labels_{region}_cleaned.csv')
            nonpaired_annotation_df = pd.read_csv(test_annotation_filename, index_col=0)

            # test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date, f'annotation_test_labels_{region}.csv')
            output_type = 'dual2'
            # decode the defect id back to defect label
            if region == 'component':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4,
                                   'oversolder': 5, 'pseudosolder': 6, 'solder_shortage': 7,
                                   'tombstone': 8, 'others': 11}  # v1.31, v1.32

                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'wrong': 3, 'undersolder': 4,
                                   'oversolder': 5, 'pseudosolder': 6, 'solder_shortage': 7,
                                   'tombstone': 8, 'others': 11}  # v1.31, v1.32
            elif region == 'package' or region == 'body':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'tombstone': 8, 'others': 11}  # v1.32
                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}  # v1.32
            elif region == 'pad':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6}  # v1.32
                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6}
            elif region == 'pins':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6,
                                   'solder_shortage': 7}

                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6,
                                   'solder_shortage': 7}

            n_class = len(defect_code)
            defect_code_link = {v: i for i, v in enumerate(defect_code.values())}
            # mode = 'ng_only'
            mode = 'flexible'
            Xtest, ytest, ytestraw, ybinary_test, material_test, test_annotation_df = generate_test_pairs(
                test_annotation_filename, defect_code=defect_code_link, region=region, mode=mode,  k_ok=1,
                replace=True, output_type=output_type)
            print(len(test_annotation_df))

            test_annotation_df = test_annotation_df.reset_index(drop=True)
            # get confidence
            confidence_lvls = []
            for i, anno in test_annotation_df.iterrows():
                insp_image = anno['insp_image']

                nonpaired_info = nonpaired_annotation_df[nonpaired_annotation_df[f'{region}_file_path_short'] == insp_image]
                cond_lvl = nonpaired_info['confidence'].values[0]
                confidence_lvls.append(cond_lvl)

            test_annotation_df['confidence'] = confidence_lvls
            print(test_annotation_df.groupby('insp_y').count())
            test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', date,
                                                   f'blackbox_test_labels_pairs_{region}_{date}_{output_type}_{mode}_230924.csv'))
            # test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', date,
            #                                        f'annotation_test_labels_pairs_{region}_{output_type}_{mode}.csv'))

    elif operation == 'rgbw_test':
        img_folder = '/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data_rgbw'
        # img_folder = '/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data_white'

        # date = '230306'
        date = '230601'

        for region in ['component','package']:
            test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date,
                                                    f'white_annotation_labels_{region}.csv')
            # test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date,
            #                                         f'white_blackbox_test_labels_{region}.csv')
            if region == 'component':
                defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'tombstone': 8, 'extra': 12}  # v1.31, v1.32

            elif region == 'package':
                defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'tombstone': 8, 'extra': 12}  # v1.32

            n_class = len(defect_code)
            defect_code_link = {v: i for i, v in enumerate(defect_code.values())}

            Xtest, ytest, ytestraw, ybinary_test, material_test, test_annotation_df = generate_test_pairs(
                test_annotation_filename, defect_code=defect_code_link, region=region, mode='flexible',
                k_ok=2)
            print(len(test_annotation_df))
            test_annotation_df = test_annotation_df.reset_index(drop=True)
            # test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', date,
            #                                        f'white_blackbox_test_labels_pairs_{region}.csv'))
            test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', date,
                                                   f'white_annotation_labels_pairs_{region}.csv'))

    elif operation == 'bbox_test_self':
        date = '230306v2'
        output_type = 'dual2'
        for region in region_list:
            test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date,
                                                    f'blackbox_test_labels_{region}_cleaned.csv')
            # decode the defect id back to defect label
            if region == 'component':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4,
                                   'oversolder': 5, 'pseudosolder': 6, 'solder_shortage': 7,
                                   'tombstone': 8, 'others': 11}  # v1.31, v1.32

                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'wrong': 3, 'undersolder': 4,
                                   'oversolder': 5, 'pseudosolder': 6, 'solder_shortage': 7,
                                   'tombstone': 8, 'others': 11}  # v1.31, v1.32
            elif region == 'package' or region == 'body':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'tombstone': 8, 'others': 11}  # v1.32
                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}  # v1.32
            elif region == 'pad':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6}  # v1.32
                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6}
            elif region == 'pins':
                if output_type == 'dual':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6,
                                   'solder_shortage': 7}

                elif output_type == 'dual2':
                    defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'undersolder': 4, 'oversolder': 5,
                                   'pseudosolder': 6,
                                   'solder_shortage': 7}

            n_class = len(defect_code)
            defect_code_link = {v: i for i, v in enumerate(defect_code.values())}

            Xtest, ytest, ytestraw, ybinary_test, material_test, test_annotation_df = generate_test_selfpairs(
                test_annotation_filename, defect_code=defect_code_link, region=region, mode='flexible', k_ok=2,
                replace=True)
            print(len(test_annotation_df))
            test_annotation_df = test_annotation_df.reset_index(drop=True)
            print(test_annotation_df.groupby('insp_y').count())
            # get certainty
            test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', date,
                                                   f'blackbox_test_labels_selfpairs_{region}_{output_type}.csv'))


    # import matplotlib.pyplot as plt
    # import cv2
    # val_frac = 0.3
    # rs_img_size = 224
    # # img_folder = '/home/robinru/shiyuan_projects/SMTdefectclassification/data/component_data/'
    # img_folder = '/home/robinru/shiyuan_projects/data/aoi_defect_data_20220906'
    # annotation_filename = os.path.join(img_folder, f'annotation_labels.csv')
    # train_annotation_df, val_annotation_df = stratified_train_val_split(annotation_filename, val_ratio=0.3,
    #                                                                     groupby='defect_label', seed=42, verbose=True)
    # Xpairs_resampled, ypair_resampled, position_resampled = mp_weighted_resampler(train_annotation_df)
    #
    # reference_annotation_df = train_annotation_df
    # # selected_id = [3, 25, 177]
    # selected_id = [1, 24, 57]
    # # check the reference ok sample for each group
    # # selected_id = None
    # if selected_id is not None:
    #     # if we know the img id to be used as reference e.g. gold template
    #     # use them directly
    #     selected_reference_df = reference_annotation_df.loc[selected_id]
    # else:
    #     # if we don't know the img id to be used as reference sample from training data with ok labels and certain labeling
    #     certainly_ok_reference_df = reference_annotation_df[
    #         (reference_annotation_df['y'] == 0) & (reference_annotation_df['certainty'] == 'sure')]
    #     selected_reference_df = certainly_ok_reference_df.groupby('position', group_keys=False).apply(
    #         lambda x: x.sample(n=3, random_state=7))
    #
    # Xpairs_resampled, ypair_resampled, position_resampled = [], [], []
    # for j in val_annotation_df.index:
    #     s2_X_path = val_annotation_df.loc[j,'X_file_path']
    #     position_chosen = val_annotation_df.loc[j, 'position']
    #     s2_y = val_annotation_df.loc[j, 'y']
    #
    #     s1_X_path = selected_reference_df.loc[selected_reference_df['position'] == position_chosen,'X_file_path'].iloc[0]
    #     s1_y = selected_reference_df.loc[selected_reference_df['position'] == position_chosen,'y'].iloc[0]
    #
    #     Xpairs_resampled.append([s1_X_path, s2_X_path])
    #     ypair_resampled.append([s1_y, s2_y])
    #     position_resampled.append(position_chosen[0])
    #
    # # visualise the reference images
    # reference_img_name_list = list(selected_reference_df['X_file_path'])
    # img_indices = list(selected_reference_df.index)
    # for j, reference_img_name in enumerate(reference_img_name_list):
    #     refFilename = os.path.join(img_folder, reference_img_name)
    #     img_sub = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    #     img_show = cv2.cvtColor(img_sub, cv2.COLOR_BGR2RGB)
    #     plt.imshow(img_show, cmap='gray')
    #     img_id = img_indices[j]
    #     plt.title(img_id)
    #     plt.show()
    # convert to pytorch dataset
    # smtdataset = ImageLoader(img_folder, Xpairs_resampled, ypair_resampled, position_resampled,
    #                       transform=transforms.Compose([transforms.Resize((rs_img_size ,rs_img_size))]))
    # data_loader = torch.utils.data.DataLoader(smtdataset, batch_size=8,
    #                                                shuffle=True, num_workers=1, pin_memory=True)
    # X1, X2, y1, y2, position = next(iter(data_loader))

