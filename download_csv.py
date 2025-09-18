import os
from train_data import get_train_csv

def down_csv(img_folder, date, part_name, version_name):
    annotation_folder = os.path.join(img_folder, 'merged_annotation', date)
    annotation_filename, val_annotation_filename, test_annotation_filename, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
            get_train_csv(annotation_folder, part_name, version_name)
    pass