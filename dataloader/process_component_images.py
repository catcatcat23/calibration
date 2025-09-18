import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.utilities import MaskOutROI

# specify image folder and annotation json folder
img_folder = '/home/robinru/shiyuan_projects/data/aoi_defect_data_20220906'
json_folder_path = '/home/robinru/shiyuan_projects/SMTdefectclassification/data/'
json_folder_name = 'defects_20220908_aug'
# json_folder_name = 'defects_20220908'
json_folder = os.path.join(json_folder_path, json_folder_name)
visualise = False
defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'oversolder': 3, 'undersolder': 4,
               'pseudosolder': 5, 'solder_ball': 6, 'tombstone': 7, 'wrong': 8, 'overturn': 9}
# file_list = ['R_2']
file_list = ['C_1','L_1','R_1', 'R_2', 'R_3', 'R_4', 'R_5', ]
alltype_annotation = []
alltype_roi_dict = []
for file in file_list:
    # loop through different json
    annotation_dic = {'position': [], 'defect_label': [], 'component_file_path': [],'certainty':[], 'package_file_path':[], 'pad_file_path':[]}
    file_path = os.path.join(json_folder, f'Defect_20220907_{file}.json')
    f_json = open(file_path)
    annotation_json = json.load(f_json)
    package_roi_list = []
    for component_json in annotation_json:
        # loop through annotation of all components
        img_name = component_json['info'].split('/')[-1]
        defect_type_annotation = component_json['labels'][0]
        assert component_json['labels'][0]['drawType'] == 'RECTANGLE'
        component_rect = [int(p+1) for p in defect_type_annotation['points']]
        defect_annotation_attr = defect_type_annotation['attr']
        defect_labels = defect_annotation_attr['defect_category']
        defect_part = defect_annotation_attr['defect_part']
        defect_certainty = defect_annotation_attr['certainty']
        designator = defect_annotation_attr['designator']
        component_category = defect_annotation_attr['comments']
        if component_category.startswith('R'):
            component_category = 'R1'
        img_file_name = os.path.join(img_folder, img_name)
        if defect_certainty == 'sure' and component_rect[3] > component_rect[1] and component_rect[2] > component_rect[0] and os.path.exists(img_file_name):
            component_xmin, component_ymin, component_xmax, component_ymax = component_rect
            if json_folder_name == 'defects_20220908':
                # if component_rect[3] > component_rect[1] and component_rect[2] > component_rect[0] and os.path.exists(img_file_name):
                    # consider the example if we are certain about the defect label
                    img_board = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
                    img_sub = img_board[component_ymin:component_ymax, component_xmin:component_xmax]

                    if img_sub.shape[0] > img_sub.shape[1]:
                        # retote the vertical image to horizontal one
                        img_sub = np.rot90(img_sub)

                    output_name = os.path.join(img_folder, f'component/component_{img_name}')
                    cv2.imwrite(output_name, img_sub)

                    if visualise:
                        img_show = cv2.cvtColor(img_sub, cv2.COLOR_BGR2RGB)
                        plt.imshow(img_show, cmap='gray')
                        plt.title(f'{img_name} \n {defect_labels}')
                        plt.show()

            elif json_folder_name == 'defects_20220908_aug':
                solder_pad_roi = []
                package_roi = []
                for li in range(1, len(component_json['labels'])):
                    roi_annotation = component_json['labels'][li]
                    roi_vertices = roi_annotation['points']
                    roi_name = roi_annotation['label']
                    roi_vertices_np = np.array(roi_vertices).reshape([4,2])
                    roi_vertices_np_clipped = np.clip(roi_vertices_np, [component_xmin, component_ymin], [component_xmax, component_ymax])
                    roi_vertices_np_shift = roi_vertices_np_clipped - np.array([component_xmin, component_ymin])
                    if roi_name == 'pad':
                        solder_pad_roi.append(roi_vertices_np_shift.astype(np.int32))
                    elif roi_name == 'package':
                        package_roi.append(roi_vertices_np_shift.astype(np.int32))
                    else:
                        print('check')

                if img_name == 'AD008C-K.001_20200921145224599_NG_COMP1060_C2_1883.png':
                    solder_pad_array = solder_pad_roi[0]
                    solder_pad_roi[0] = np.hstack([solder_pad_array[:,1:2], solder_pad_array[:,0:1]])
                elif img_name == 'AD008C-K.001_20200921145224599_NG_COMP1060_C2_1613.png':
                    solder_pad_array = solder_pad_roi[1]
                    solder_pad_roi[1] = np.hstack([solder_pad_array[:, 1:2], solder_pad_array[:, 0:1]])

                cropped_image_name = os.path.join(img_folder, f'component/component_{img_name}')
                img = cv2.imread(cropped_image_name, cv2.IMREAD_COLOR)
                img_pad, mask_pad = MaskOutROI(img.copy(), solder_pad_roi, value_scale=1)
                output_name_pad = os.path.join(img_folder, f'pad/pad_{img_name}')
                cv2.imwrite(output_name_pad, img_pad)

                if len(package_roi) == 0:
                    img_package = np.zeros_like(img.copy(), dtype=np.int32)
                else:
                    package_roi_list.append(package_roi)
                    img_package, mask_package = MaskOutROI(img, package_roi, value_scale=1)
                output_name_package = os.path.join(img_folder, f'package/package_{img_name}')
                cv2.imwrite(output_name_package, img_package)

                if visualise:
                    img_show = img.copy()
                    cv2.drawContours(img_show, package_roi, 0, (0, 255, 0), 1)

                    cv2.drawContours(img_show, solder_pad_roi, -1, (0, 0, 255), 1)

                    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
                    # plt.imshow(img_show, cmap='gray')
                    # plt.title(f'{img_name} \n {defect_labels}')
                    # plt.show()

                    plt.imshow(img_pad, cmap='gray')
                    plt.title(f'{img_name} \n {defect_labels} pad')
                    plt.show()



            annotation_dic['position'].append(component_category)
            annotation_dic['defect_label'].append(defect_labels[-1].lower())
            annotation_dic['component_file_path'].append(f'component_{img_name}')
            annotation_dic['certainty'].append(defect_certainty)
            annotation_dic['package_file_path'].append(f'package_{img_name}')
            annotation_dic['pad_file_path'].append(f'pad_{img_name}')

    annotation_pd = pd.DataFrame(annotation_dic)
    alltype_annotation.append(annotation_pd)
    # print(annotation_pd.groupby('defect_label').count())

alltype_annotation_df = pd.concat(alltype_annotation)
alltype_annotation_df.reset_index(inplace=True, drop=True)
# set defect labels to numerical number
alltype_annotation_df['y'] = [defect_code[d_type] for d_type in alltype_annotation_df['defect_label']]
csvFilename = os.path.join(img_folder, f'annotation_labels.csv')
alltype_annotation_df.to_csv(csvFilename)
print(alltype_annotation_df.groupby('defect_label').count())