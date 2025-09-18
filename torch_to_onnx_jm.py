# this script convert torch model to onnx based on code from https://github.com/sohaib023/siamese-pytorch/blob/master/torch_to_onnx.py
import numpy as np
import os
import argparse
import onnx
import torch
from models.MPB3 import MPB3net
from models.MPB3_sp import MPB3netSP
import onnxruntime as ort
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained = False
rs_img_size = 224
# ckp_folder = './models/checkpoints_772'
# ckp_folder = '/mnt/pvc-nfs-dynamic/robinru/model_ckps/'
# ckp_folder = '/mnt/pvc-nfs-dynamic/robinru/model_ckps/aoi-sdkv0.10.0'
v_folder = 'singlepinpad/v2.17.6tp1'
ckp_folder = f'/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/models/checkpoints/{v_folder}'
version_folder = 'singlepinpadv2.17.6tp1select'
# version_folder = 'singlepinpadv2.3.0'
# image normalization mean and std
batch_size = 8
gpu_exists = True
seed = 42
                  
ckp_name_list = ['singlepinpadmobilenetv3smallrs12832s42c3val0.15b256_ckp_bestv2.17.6f1certaincp0520.0j0.4lr0.025nb256nm256dual2bestbiacc']
output_type = 'dual2'
region_list = ['singlepinpad']

for region, ckp_name in zip(region_list, ckp_name_list):
    n_class = int(ckp_name.split('val')[0].split(f's{seed}c')[-1])
    if region == 'singlepinpad':
        n_units = [256, 256]
        rs_img_size_h, rs_img_size_w = 32, 128
    elif region == 'singlepad':
        n_units = [256, 256]
        rs_img_size_h, rs_img_size_w = 64, 64
    elif region == 'body':
        n_units = [256, 256]
        rs_img_size_h, rs_img_size_w = 224, 224
    else:
        if region == 'pins_part':
            rs_img_size_h, rs_img_size_w = 112, 224
        elif region == 'padgroup':
            rs_img_size_h, rs_img_size_w = 128, 128
        else:
            rs_img_size_h, rs_img_size_w = 224, 224

        if 'top' in ckp_name:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                       ckp_name.split('f16')[-1].split('top')[0].split('nb')[-1].split('nm')]
        elif 'last' in ckp_name:
            n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                       ckp_name.split('f16')[-1].split('last')[0].split('nb')[-1].split('nm')]
        else:
            n_units = [128, 128]

    backbone_arch = ckp_name.split('rs')[0].split(region)[-1]
    pretrained = 'pretrained' in backbone_arch
    torch.manual_seed(seed)

    torch_ckp_path = os.path.join(ckp_folder, version_folder,
                                  f'{ckp_name}.pth.tar')
    if 'CLL' in ckp_name:
        output_type = 'dual2CLL'
    elif 'CL' in ckp_name:
        output_type = 'dual2CL'

    if 'cbam' in backbone_arch and 'CL' in output_type:
        print(backbone_arch)
        if region == 'body' or region == 'component':
            attention_list_str = backbone_arch.split('cbam')[-1]
            if len(attention_list_str.split('cbam')[-1]) == 0:
                attention_list = None
            else:
                attention_list = [int(attention_list_str)]
        
        from models.MPB3_attn_ConstLearning_test import MPB3net
        print(attention_list)  

        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                            output_form=output_type, attention_list=attention_list)
    elif 'cbam' in backbone_arch:
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
    elif 'CL' in output_type:
        from models.MPB3_ConstLearning_test import MPB3net

        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                        output_form=output_type)

    else:
        from models.MPB3 import MPB3net

        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                        output_form=output_type)
        
    print(f'=> Loading checkpoint {torch_ckp_path}')
    checkpoint = torch.load(torch_ckp_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'=> Loaded checkpoint {torch_ckp_path}')
    model.to(device)
    model.eval()

    dummy_inputs = (torch.rand(1, 3, rs_img_size_h, rs_img_size_w).to(device),
                   torch.rand(1, 3, rs_img_size_h, rs_img_size_w).to(device))
    # compute the torch model outputs at test inputs
    torch_outputs = model(dummy_inputs[0], dummy_inputs[1])
    torch_output1, torch_output2 = torch_outputs[0].detach().cpu().numpy(), torch_outputs[1].detach().cpu().numpy()
    print(f'torch outputs= {torch_outputs}')

    # export the model to onnix
    ckp_rename = ckp_name.replace('v2.10.30', 'v2.4.0')
    onnx_output_path = os.path.join(ckp_folder, version_folder, f'{ckp_rename}.onnx')

    input_names = ['input1', 'input2']
    output_names = ['output1', 'output2']
    dynamic_axes = {
        input_names[0]: {
            0: 'batch',
            # 2: 'height',
            # 3: 'width'
        },
        input_names[1]: {
            0: 'batch',
            # 2: 'height',
            # 3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
            2: 'num_classes',
        },
    }

    torch.onnx.export(model, dummy_inputs, onnx_output_path,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes= dynamic_axes,
                      export_params=True)

    # verify the onnx model using ONNX library
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph)) # todo gives error

    # run the onnx model with one of the runtimes that support ONNX
    ort_session = ort.InferenceSession(onnx_output_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime outputs at test inputs
    onnx_inputs = {ort_session.get_inputs()[i].name: to_numpy(di) for i, di in enumerate(dummy_inputs)}
    onnx_outputs = ort_session.run(None, onnx_inputs)

    print(f'onnix outputs= {onnx_outputs}')
