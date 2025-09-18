import os
import cv2
import torch
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.utilities import TransformImage, split
# from scipy.special import softmax
from copy import deepcopy

class GradCAM():
    def __init__(self, model, target_module, target_layer):
        self.model = model
        self.model.eval()
        self.model.cuda()

        module = getattr(self.model, target_module)
        getattr(module, target_layer).register_backward_hook(self._backward_hook)
        getattr(module, target_layer).register_forward_hook(self._forward_hook)

    def forward(self, ref_img, insp_img, model_path, cam_type = 'bos'):
        self.grads = []
        self.fmaps = []
        if "CL" in model_path:
            output_bos, output_bom, _, _ = self.model(ref_img, insp_img)
        else:
            output_bos, output_bom = self.model(ref_img, insp_img)

        # apply softmax
        output_bos_np_softmax = F.softmax(output_bos, dim=1)
        output_bom_np_softmax = F.softmax(output_bom, dim=1)
        self.cam_type = cam_type
        if cam_type == 'bos':
            print(f"compute binary GradCAM")
            cls_idx  = torch.argmax(output_bos_np_softmax).item()
            score = output_bos_np_softmax[:, cls_idx].sum()
        elif cam_type == 'bom':
            print(f"compute multi-class GradCAM")
            cls_idx  = torch.argmax(output_bom_np_softmax).item()
            score = output_bom_np_softmax[:, cls_idx].sum()
        score.backward()

        target_fmap = self.fmaps[0][0].clone().detach().cpu().numpy() # 第一个是insp
        target_grad = self.grads[1][0].clone().detach().cpu().numpy().squeeze(0)
        target_img = insp_img.clone().detach().cpu().numpy().squeeze(0)
        self.insp_cam_mask = self._compute_cam_mask(target_img, target_fmap, target_grad)
        
        target_fmap = self.fmaps[1][0].clone().detach().cpu().numpy() # 第二个特征是ref
        target_grad = self.grads[0][0].clone().detach().cpu().numpy().squeeze(0)
        target_img = ref_img.clone().detach().cpu().numpy().squeeze(0)
        self.ref_cam_mask = self._compute_cam_mask(target_img, target_fmap, target_grad)        

    def show_cam(self, ori_img, save_cam_path, cam_target = 'insp'):
        self._compute_cam(ori_img, cam_target)
        cv2.imwrite(save_cam_path, self.cam)
        # pass

    # axes[1].set_title(f'Insp Mclass = {insp_label} \n ({certainty})')
    def show_contrastive_results(self, ref_img, insp_img, rel_insp_label, predict_insp_label, ori_model_predict_result, save_path):
        self._compute_cam(insp_img, 'insp')
        insp_cam = deepcopy(self.cam)
        self._compute_cam(ref_img, 'ref')
        ref_cam = deepcopy(self.cam)

        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        insp_img = cv2.cvtColor(insp_img, cv2.COLOR_BGR2RGB)
        insp_cam = cv2.cvtColor(insp_cam, cv2.COLOR_BGR2RGB)
        ref_cam = cv2.cvtColor(ref_cam, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10)) 
        axs[0, 0].imshow(ref_img)
        axs[0, 0].set_title('ref_img')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(insp_img)
        axs[0, 1].set_title(f'insp_img \n real mclass: {rel_insp_label}')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(ref_cam)
        axs[1, 0].set_title('ref_cam')
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(insp_cam)
        axs[1, 1].set_title(f'insp_{self.cam_type}_cam \n final pred Mclass = {predict_insp_label} \n ori pred = ({ori_model_predict_result})')
        axs[1, 1].axis('off')

        plt.savefig(save_path)
        plt.close()

    def _backward_hook(self, model, input_grad, output_grad):
        # print(f"input_grad shape: {input_grad.shape}")
        # print(f"input_grad: {input_grad}")
        # print(f"output_grad: {output_grad}")
        self.grads.append(output_grad)
  
    def _forward_hook(self, model, input, output):
        # print(f"input_type: {type(input)}")
        # print(f"input: {input}")
        self.fmaps.append(output) # forward了两次，第一次是insp，第二次是ref

    def _compute_cam_mask(self, img, fmap, grad_map):
        img_W, img_H = img.shape[1:]

        alpha = np.mean(grad_map, axis=(1, 2))  # GAP
        # for k, ak in enumerate(alpha):
        #     cam += ak * fmap[k]  # linear combination
        activation = np.tensordot(alpha, fmap, axes=(0, 0))
        activation_min = np.min(activation)
        activation_offset = activation - activation_min
        activation_offset_max = np.max(activation_offset)
        activation_normal = activation_offset / activation_offset_max

        cam_mask = cv2.resize(activation_normal, (img_W, img_H))
        return cam_mask
    
    def _compute_cam(self, ori_img, cam_target = 'insp'):
        ori_H, ori_W, _ = ori_img.shape 
        if cam_target == 'insp':
            cam_mask = self.insp_cam_mask
        elif cam_target == 'ref':
            cam_mask = self.ref_cam_mask

        cam_mask = cv2.resize(cam_mask, (ori_W, ori_H))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET) / 255
        img_nor = ori_img / 255

        cam = heatmap + img_nor
        cam = cam / np.max(cam)
        self.cam = np.uint8(255 * cam)
 
        

def init_model(model_path, backbone_arch, region, n_class, n_units):
    if 'CLL' in model_path:
        output_type = "CLLdual2"
    elif 'CL' in model_path:
        output_type = "CLdual2"
    else:
        output_type = "dual2"


    # define model and load check point
    # if 'resnetsp' in backbone_arch:
    #     from models.MPB3 import MPB3net
    if 'cbam' in backbone_arch and 'CL' not in model_path:
        print(backbone_arch)
        if region == 'body' or region == 'component':
            attention_list_str = backbone_arch.split('cbam')[-1]
            if len(attention_list_str.split('cbam')[-1]) == 0:
                attention_list = None
            else:
                attention_list = [int(attention_list_str)]
        
        from models.MPB3_attention import MPB3net
        print(attention_list)

        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                            output_form=output_type, attention_list=attention_list)
    elif 'cbam' in backbone_arch and 'CL' in model_path:
        print(backbone_arch)
        if region == 'body' or region == 'component':
            attention_list_str = backbone_arch.split('cbam')[-1]
            if len(attention_list_str.split('cbam')[-1]) == 0:
                attention_list = None
            else:
                attention_list = [int(attention_list_str)]
        
        from models.MPB3_attention_CL import MPB3net
        print(attention_list)

        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                            output_form=output_type, attention_list=attention_list)           
    
    elif 'CL' in model_path:
        from models.MPB3_CL import MPB3net
        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units, output_form=output_type)
    else:
        from models.MPB3 import MPB3net

        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                        output_form=output_type)
    
    return model

def parser_model_param(model_path, region):
    backbone_arch = model_path.split('rs')[0].split(region)[-1]
    if 'nb' not in model_path:
        n_units = [128, 128]
    else:
        if 'top' in model_path and 'Cosin' in model_path:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        model_path.split('f16')[-1].split('top')[0].split('nb')[-1].split('Cosin')[0].split('nm')]
        elif 'top' in model_path and 'CL' in model_path:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        model_path.split('f16')[-1].split('top')[0].split('nb')[-1].split('CL')[0].split('nm')]
        elif 'top' in model_path:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        model_path.split('f16')[-1].split('top')[0].split('nb')[-1].split('nm')]
        elif 'Cosin' in model_path:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        model_path.split('f16')[-1].split('last')[0].split('nb')[-1].split('Cosin')[0].split('nm')]
        elif 'CL' in model_path:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        model_path.split('f16')[-1].split('last')[0].split('nb')[-1].split('CL')[0].split('nm')]
        else:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        model_path.split('f16')[-1].split('last')[0].split('nb')[-1].split('nm')]
    
    defect_code, n_class, rs_img_size_w, rs_img_size_h = init_defect_code(region, model_path)
    defect_decode = {i: k for i, k in enumerate(defect_code.keys())}
    defect_decode.update({-1: 'ng'})

    return backbone_arch, n_units, defect_code, defect_decode, n_class, rs_img_size_w, rs_img_size_h

def init_defect_code(region, model_path):
    if region == 'component':
        rs_img_size_w = 224
        rs_img_size_h = 224
        rsstr = f'rs{rs_img_size_w}'

        if 'fly' in model_path:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11, 'fly': 12}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4, 'tombstone': 5, 'others': 6, 'fly': 7}
            defect_code_slim = {'ok': 0, 'missing': 1,'tombstone': 8, 'others': 11,  'fly': 12}
#             defect_code_slim = {'ok': 0, 'missing': 1, 'wrong': 3,'tombstone': 8, 'others': 11}
            attention_list = None
            if 'flycbam1' in model_path:
                attention_list_str = model_path.split('flycbam')[-1]
                attention_list = [int(attention_list_str)]

        else:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4, 'tombstone': 5, 'others': 6}
            defect_code_slim = {'ok': 0, 'missing': 1,'tombstone': 8, 'others': 11}

    elif region == 'body' or region == 'bodyl':
        rs_img_size_w = 224
        rs_img_size_h = 224
        rsstr = f'rs{rs_img_size_w}'

        if 'slim' in model_path:
#             defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
            defect_code = {'ok': 0, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'wrong': 4}
        else:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4}
#             defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
#             defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4, 'tombstone': 5, 'others': 6}
#             defect_code_slim = {'ok': 0, 'missing': 1, 'wrong': 3}            

    elif region == 'single_pin' or region == 'singlepinpad':
        rs_img_size_w = 128
        rs_img_size_h = 32
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
            
        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'singlepad':
        rs_img_size_w = 64
        rs_img_size_h = 64
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'pins_part' or region == 'padgroup':
        rs_img_size_w, rs_img_size_h = 128, 128
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'
        defect_code_slim = {'ok': 0, 'solder_shortage': 7}
        if rsstr == f'rs224112':
            defect_code = {'ok': 0, 'missing': 1, 'undersolder': 4, 'pseudosolder': 6, 'solder_shortage': 7}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'undersolder': 4, 'pseudosolder': 5, 'solder_shortage': 6}  # v1.32
        elif 'nonslim' in model_path:
            defect_code = {'ok': 0, 'missing': 1, 'solder_shortage': 7}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'solder_shortage': 4}  # v1.32
        else:
            defect_code = {'ok': 0, 'solder_shortage': 7}
            defect_sdk_decode = {'ok': 2, 'solder_shortage': 3}  # v1.32

    n_class = len(defect_code)

    return defect_code, n_class, rs_img_size_w, rs_img_size_h

def load_data(data_path, csv_path, date, rs_img_size_h, rs_img_size_w, batch_size = 256):
    val_image_pair_path_list = [os.path.join(data_path, 'merged_annotation', date, f'{csv_path}')]
    val_image_pair_data_list = [pd.read_csv(path, index_col=0) for path in val_image_pair_path_list]
    val_image_pair_data_raw = pd.concat(val_image_pair_data_list).reset_index(drop=True)

    val_image_pair_data = val_image_pair_data_raw.drop_duplicates(subset=['ref_image', 'insp_image']).reset_index(drop=True)
    val_image_pair_res = val_image_pair_data[['ref_image', 'insp_image']].copy()
    binary_label_list = val_image_pair_data['binary_y'].tolist()
    ref_image_list = []
    insp_image_list = []
    ref_label_list = []
    insp_label_list = []
    ref_image_name_list = []
    insp_image_name_list = []
    counter = 0
    ref_image_batches = []
    insp_image_batches = []
    ref_image_path_list = []
    insp_image_path_list = []
    n_batches = int(len(val_image_pair_data) / batch_size)
    if len(val_image_pair_data) < batch_size:
        batch_idices_list = [range(len(val_image_pair_data))]
        n_batches = 1
    else:
        batch_idices_list = split(list(val_image_pair_data.index), n_batches)
    for batch_indices in batch_idices_list:

        ref_batch = []
        insp_batch = []
        for i in batch_indices:

            if counter % 1000 == 0:
                print(f'{counter}')
            counter += 1
            val_image_pair_i = val_image_pair_data.iloc[i]

            ref_image_path = os.path.join(data_path, val_image_pair_i['ref_image'])
            insp_image_path = os.path.join(data_path, val_image_pair_i['insp_image'])

            ref_image_path_list.append(ref_image_path)
            insp_image_path_list.append(insp_image_path)

            ref_image_name = val_image_pair_i['ref_image'].split('/')[-1]
            insp_image_name = val_image_pair_i['insp_image'].split('/')[-1]
            # scale, normalize and resize test images
            ref_image = TransformImage(img_path=ref_image_path, rs_img_size_h=rs_img_size_h,
                                       rs_img_size_w=rs_img_size_w,
                                      ).transform()  # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
            insp_image = TransformImage(img_path=insp_image_path, rs_img_size_h=rs_img_size_h,
                                        rs_img_size_w=rs_img_size_w,
                                        ).transform() # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p

            # get ground truth labels
            ref_label = val_image_pair_i['ref_y']
            insp_label = val_image_pair_i['insp_y']
            ref_batch.append(ref_image)
            insp_batch.append(insp_image)

            ref_label_list.append(ref_label)
            insp_label_list.append(insp_label)
            ref_image_name_list.append(ref_image_name)
            insp_image_name_list.append(insp_image_name)

        ref_image_batches.append(np.concatenate(ref_batch, axis=0))
        insp_image_batches.append(np.concatenate(insp_batch, axis=0))

    n_batches = np.max([int(len(ref_image_list) / batch_size), 1])

    binary_labels = np.array(binary_label_list).reshape([-1, 1])
    ref_labels = np.array(ref_label_list).reshape([-1, 1])
    insp_labels = np.array(insp_label_list).reshape([-1, 1])

    return ref_image_batches, insp_image_batches, ref_image_path_list, insp_image_path_list, ref_labels, insp_labels, binary_labels

def test_gradCAM(args, model_path, model, gradCAMer, ref_image_batches, insp_image_batches, ref_image_path_list, insp_image_path_list):
    output_bos_list = []
    output_bom_list = []
    for ref_img, insp_img, ref_image_path, insp_image_path in zip(ref_image_batches, insp_image_batches, ref_image_path_list, insp_image_path_list):
        ref_img, insp_img = torch.FloatTensor(ref_img).cuda(), torch.FloatTensor(insp_img).cuda()

        ori_ref_img = cv2.imread(ref_image_path)
        ori_insp_img = cv2.imread(insp_image_path)

        # ori_ref_img_size = ori_ref_img.shape[:-1]
        # ori_insp_img_size = ori_insp_img.shape[:-1]
        gradCAMer.forward(ref_img, insp_img, model_path, args.cam_type)

        # if args.cam_target == 'insp':
        #     cam_img_path = os.path.join(args.cam_save_path, os.path.basename(insp_image_path))
        #     gradCAMer.show_cam(ori_insp_img, cam_img_path)

        # elif args.cam_target == 'ref':
        #     cam_img_path = os.path.join(args.cam_save_path, os.path.basename(ref_image_path))
        #     gradCAMer.show_cam(ori_ref_img, cam_img_path)

        cam_contrastive_path = os.path.join(args.cam_save_path, os.path.basename(insp_image_path))
        gradCAMer.show_contrastive_results(ori_ref_img, ori_insp_img, cam_contrastive_path)

def GradCAMer(args):
    model_path = args.model_path
    backbone_arch, n_units, defect_code, n_class, rs_img_size_w, rs_img_size_h = parser_model_param(model_path, args.region)

    model = init_model(model_path, backbone_arch, args.region, n_class, n_units)

    print(f'=> Loading checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    ckp_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'=> Loaded checkpoint {model_path} {ckp_epoch}')

    gradCAMer = GradCAM(model, args.target_module, args.target_layer)
    ref_image_batches, insp_image_batches, ref_image_path_list, insp_image_path_list, ref_labels, insp_labels, binary_labels \
                = load_data(args.data_path, args.csv_path, args.date, rs_img_size_h, rs_img_size_w, batch_size = 1)
    
    test_gradCAM(args, model_path, model, gradCAMer, ref_image_batches, insp_image_batches, ref_image_path_list, insp_image_path_list)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='component', help = '缺陷类型')
    parser.add_argument('--model_path', type=str, 
                        default='./models/checkpoints/componentv4.6tmptselect/componentfcdropoutmobilenetv3largecbam13rs224s42c5val0.0_ckp_bestv4.6.0e250CLLwhite20.0f16j0.4lr0.025nb256nm256dual2CLLtop0.pth.tar', 
                        help = '')

    parser.add_argument('--target_module', type=str, default='cnn_encoder', help = '要进行gradcam的层')
    parser.add_argument('--target_layer', type=str, default='16', help = '要进行gradcam的层')
    parser.add_argument('--cam_type', type=str, default='bom', help = '对二分类还是多分类结果计算gradcam')
    parser.add_argument('--cam_target', type=str, default='insp', help = '对检测板还是金板计算gradcam') 
    parser.add_argument('--cam_save_path', type=str, default='./GradCAM_Results/{region}_{target_module}_{target_layer}_{cam_type}_{cam_target}', 
                        help = '对检测板还是金板计算gradcam')    


    parser.add_argument('--date', type=str, default='240913_white', help = 'csv文件所在文件夹名称')
    parser.add_argument('--data_path', type=str, default='/mnt/dataset/xianjianming/data_clean_white/', help = '数据路径')
    parser.add_argument('--csv_path', type=str, default='aug_test_pair_labels_component_241223_final_white_DA574.csv', help = 'csv路径')

    args = parser.parse_args()
    args.cam_save_path = args.cam_save_path.format(
        region = args.region,
        target_module = args.target_module,
        target_layer = args.target_layer,
        cam_type = args.cam_type,
        cam_target = args.cam_target,
    )
    os.makedirs(args.cam_save_path, exist_ok=True)

    GradCAMer(args)
    pass