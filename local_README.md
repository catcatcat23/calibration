# SMTdefect_classification

This is the code repository for performing SMT defect classification based on a [multiple position-based 
bi-branch (MPB3)](https://link.springer.com/article/10.1007/s10845-021-01870-4) neural network. 
MPB3 takes in two input images like Simamese network: an ok sample ideally 
from gold board and a test sample (ok/ng) from inspected board. MPB3 provides two outputs: a binary output quantifying the 
similarity between the ok sample and test sample, and another multi-class output informing the exact defect class.
A multi-position weighted-resampling (MPWR) method is used to prepare paired data and balance class imbalance.
Note 'position' here has the same meaning as the component model series in our setting.

## Dependencies

To install the following dependencies:
 - Python >= 3.6.0
 - torch=1.10.1
 - torchvision=0.11.2
 - opencv-python=4.6.0
 - matplotlib
 - pandas
 - shutil

## Download data

Download defect component data from [robin_own_cloud](http://10.8.0.33:8080/s/ZOSHDbfkWn0JsJO) and put it under your directory 
e.g `/home/robinru/shiyuan_projects/data/aoi_defect_data_20220906`. There exists a `annotation_labels.csv`
which specifies the annotation information.

## Train the MPB3 network on defect data

Use `resnet18` as the backbone and train the MPB3 network from scratch
```
python main.py --arch=resnet18 --mode=train -lr=0.001
```

Use pretrained `resnet18` as the backbone and train the MPB3 network for fine-tuning
```
python main.py --arch=resnet18_pretrained --mode=train -lr=0.0001
```

Checkpoints would be saved under the directory specified by `--ckp=`

## Validate the trained MPB3 network on validation data 

Load the [checkpoints](http://10.8.0.33:8080/s/DtgV1RKmV55XUty) of the trained MPB3 network and run inference on validation defect data of entire component image `region=component`.
```
python main.py --arch=resnet18 --mode=val --region=component
```
Validation image pairs would be plotted; each contains an ok reference sample (left) and a ok/ng validation sample (right).

## Run inference with trained MPB3 network 
Download the test data from [robin_own_cloud](http://10.8.0.33:8080/s/ZOSHDbfkWn0JsJO) and put it under the directory 
`./data/defects_inference_tests_region`. Then specify whether to run the model in `onnx` or `torch`
```
python inference_main.py
```