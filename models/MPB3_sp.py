import torch
import torch.nn as nn
from models.resnet_sp import resnet18, resnet34, resnet50, resnet101, resnetsp18, resnetsp34, resnetsp50
import torchvision
from models.utilities import SEBasicBlock, SELayer

class MPB3netSP(nn.Module):

    def __init__(self, backbone='resnet18', r=16, pretrained=False, n_class=4, n_units=[256, 256], output_form='dual2'):
        super(MPB3netSP, self).__init__()

        if 'resnet18' in backbone:
            feature_extractor = resnet18(pretrained=pretrained)
        elif 'resnetsp18' in backbone:
            feature_extractor = resnetsp18()
        elif 'resnetsp34' in backbone:
            feature_extractor = resnetsp34()
        elif 'resnetsp50' in backbone:
            feature_extractor = resnetsp50()
        elif 'resnet34' in backbone:
            feature_extractor = resnet34(pretrained=pretrained)
        elif 'resnet50' in backbone:
            feature_extractor = resnet50(pretrained=pretrained)
        elif 'resnet101' in backbone:
            feature_extractor = resnet101(pretrained=pretrained)

        elif 'mobilenetv3small' in backbone:
            feature_extractor = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        elif 'mobilenetv3large' in backbone:
            feature_extractor = torchvision.models.mobilenet_v3_large(pretrained=pretrained, width_mult=1.0,  reduced_tail=False, dilated=False)
        elif 'mobilenetv3quant' in backbone:
            feature_extractor = torchvision.models.quantization.mobilenet_v3_large(pretrained=pretrained)

        else:
            print('not implemented')

        self.output_form = output_form
        bos_fc_out = [n_units[0], 2]
        bom_fc_out = [n_units[1], n_class]
        if self.output_form == 'dual':
            bos_c_multiplier = 2
            bom_c_multiplier = 1
        elif self.output_form == 'dual2':
            bos_c_multiplier = 2
            bom_c_multiplier = 2
        elif self.output_form == 'mclass':
            bos_c_multiplier = 2
            bom_c_multiplier = 2
        else:
            print('Not implemented')
            assert False
        if backbone.startswith('sepost'):
            if 'resnet' in backbone:
                if 'resnetsp' in backbone:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.layer1,
                                                     feature_extractor.layer2, feature_extractor.layer3,
                                                     feature_extractor.layer4
                                                     )
                else:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.maxpool,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             # feature_extractor.avgpool
                                             )

                self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, 
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             # resnet.avgpool
                                             )
                feature_channel = self.cnn_encoder[-1][-1].conv1.weight.shape[1]
            elif 'mobilenet' in backbone or 'efficientnet' in backbone:
                self.cnn_encoder = feature_extractor.features
                feature_channel = feature_extractor.features[-1].out_channels

            se_attention_bos = SEBasicBlock(int(feature_channel * bos_c_multiplier), int(feature_channel * bos_c_multiplier), reduction=r)
            se_attention_bom = SEBasicBlock(int(feature_channel * bom_c_multiplier), int(feature_channel * bom_c_multiplier), reduction=r)

            # branch of similarity
            self.head_bos = nn.Sequential(
                se_attention_bos,
                nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                nn.Flatten(),
                nn.BatchNorm1d(int(feature_channel * bos_c_multiplier)),
                nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                nn.BatchNorm1d(bos_fc_out[0]),
                nn.ReLU(inplace=True),
                nn.Linear(bos_fc_out[0], bos_fc_out[1])
                # nn.Sigmoid()
            )
            # branch of multi-classification
            self.head_bom = nn.Sequential(
                se_attention_bom,
                nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                nn.Flatten(),
                nn.BatchNorm1d(int(feature_channel * bom_c_multiplier)),
                nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                nn.BatchNorm1d(bom_fc_out[0]),
                nn.ReLU(inplace=True),
                nn.Linear(bom_fc_out[0], bom_fc_out[1]))

        elif backbone.startswith('fcdropout'):
            if 'resnet' in backbone:
                if 'resnetsp' in backbone:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.layer1,
                                                     feature_extractor.layer2, feature_extractor.layer3,
                                                     feature_extractor.layer4, feature_extractor.maxpool
                                                     )
                else:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.maxpool,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             feature_extractor.avgpool
                                             )
                # self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu,
                #                                  feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                #                                  feature_extractor.avgpool
                #                                  )
                feature_channel = self.cnn_encoder[-2][-1].conv1.weight.shape[1]

                # branch of similarity
                self.head_bos = nn.Sequential(
#                     nn.AdaptiveMaxPool2d((1,1)),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.BatchNorm1d(int(feature_channel * bos_c_multiplier)),
                    nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(bos_fc_out[0], bos_fc_out[1])
                )
                # branch of multi-classification
                self.head_bom = nn.Sequential(
#                     nn.AdaptiveMaxPool2d((1,1)),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.BatchNorm1d(int(feature_channel * bom_c_multiplier)),
                    nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(bom_fc_out[0], bom_fc_out[1]))

            elif 'mobilenet' in backbone:
                self.cnn_encoder = nn.Sequential(feature_extractor.features, feature_extractor.avgpool, nn.Flatten())
                feature_channel = feature_extractor.features[-1].out_channels
                # branch of similarity
                self.head_bos = nn.Sequential(
                    nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bos_fc_out[0], bos_fc_out[1]))

                # branch of multi-classification
                self.head_bom = nn.Sequential(
                    nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bom_fc_out[0], bom_fc_out[1]))


        else:
            if 'resnet' in backbone:
                if 'resnetsp' in backbone:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.layer1,
                                                     feature_extractor.layer2, feature_extractor.layer3,
                                                     feature_extractor.layer4, feature_extractor.maxpool
                                                     )
                else:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.maxpool,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             feature_extractor.avgpool
                                             )
                feature_channel = self.cnn_encoder[-2][-1].conv1.weight.shape[1]

                se_attention_bos = SELayer(int(feature_channel * bos_c_multiplier), r=r)
                se_attention_bom = SELayer(int(feature_channel * bom_c_multiplier), r=r)

                # branch of similarity
                self.head_bos = nn.Sequential(
                                              se_attention_bos,
                                              nn.AdaptiveMaxPool2d((1,1)),
#                                               nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                                              nn.Flatten(),
                                              nn.BatchNorm1d(int(feature_channel*bos_c_multiplier)),
                                              nn.Linear(int(feature_channel* bos_c_multiplier), bos_fc_out[0]),
                                              nn.BatchNorm1d(bos_fc_out[0]),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(bos_fc_out[0], bos_fc_out[1])
                                              # nn.Sigmoid()
                )
                # branch of multi-classification
                self.head_bom = nn.Sequential(
                                              se_attention_bom,
                                              nn.AdaptiveMaxPool2d((1,1)),
#                                               nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                                              nn.Flatten(),
                                              nn.BatchNorm1d(int(feature_channel * bom_c_multiplier)),
                                              nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                                              nn.BatchNorm1d(bom_fc_out[0]),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(bom_fc_out[0], bom_fc_out[1]))

            elif 'mobilenet' in backbone:
                self.cnn_encoder = feature_extractor.features
                feature_channel = feature_extractor.features[-1].out_channels

                se_attention_bos = SELayer(int(feature_channel * bos_c_multiplier), r=r)
                se_attention_bom = SELayer(int(feature_channel * bom_c_multiplier), r=r)

                # branch of similarity
                self.head_bos = nn.Sequential(
                    se_attention_bos,
                    nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                    nn.Flatten(),
                    nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bos_fc_out[0], bos_fc_out[1])
                    # nn.Sigmoid()
                )
                # branch of multi-classification
                self.head_bom = nn.Sequential(
                    se_attention_bom,
                    nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                    nn.Flatten(),
                    nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bom_fc_out[0], bom_fc_out[1]))

    def forward(self, x_1, x_2=None):
        if x_2 is None and x_1.shape[0]==2:# [x1,x2]
            x_2 = x_1[1]
            x_1 = x_1[0]
        elif x_2 is None:# torch.concat([x1,x2])
#             a = 1
            x_2 = x_1[128:]
            x_1 = x_1[:128]

        # x_1 is always ok sample
        # x_2 can be ok or ng sample

        # encoder generate feature embeddings of dimension (H,W,M)
        if self.output_form == 'dual':
            feature_2 = self.cnn_encoder(x_2)
            # print(feature_2.shape)
            if x_1 is not None:
                feature_1 = self.cnn_encoder(x_1)
                # predict with branch of similarity: binary output (dimension=2)
                feature_concat = torch.cat([feature_1, feature_2], dim=1)
                logits_output_bos = self.head_bos(feature_concat)
            else:
                logits_output_bos = None
            # predict with branch of multi-classification: K-class output (dimension=K)
            logits_output_bom = self.head_bom(feature_2)
        elif self.output_form == 'dual2':
            feature_2 = self.cnn_encoder(x_2)
            feature_1 = self.cnn_encoder(x_1)
#             feature_all = self.cnn_encoder(x_1)
#             feature_1 = feature_all[:int(x_1.shape[0]/2)]
#             feature_2 = feature_all[int(x_1.shape[0]/2):]
            feature_concat = torch.cat([feature_1, feature_2], dim=1)

            # predict with branch of similarity: binary output (dimension=2)
            logits_output_bos = self.head_bos(feature_concat)
            # predict with branch of multi-classification: K-class output (dimension=K)
            logits_output_bom = self.head_bom(feature_concat)

        elif self.output_form == 'mclass':
            feature_2 = self.cnn_encoder(x_2)
            # print(feature_2.shape)
            feature_1 = self.cnn_encoder(x_1)
            # predict with branch of similarity: binary output (dimension=2)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)
            logits_output_bos = None

            # predict with branch of multi-classification: K-class output (dimension=K)
            logits_output_bom = self.head_bom(feature_concat)

        else:
            print('Not implemented')
            assert False

        return logits_output_bos, logits_output_bom

    def shap_bos(self, x_1, x_2):
        # x_1 is always ok sample
        # x_2 can be ok or ng sample

        # encoder generate feature embeddings of dimension (H,W,M)
        feature_2 = self.cnn_encoder(x_2)
        # print(feature_2.shape)
        if x_1 is not None:
            feature_1 = self.cnn_encoder(x_1)
            # predict with branch of similarity: binary output (dimension=2)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)
            logits_output_bos = self.head_bos(feature_concat)
        else:
            logits_output_bos = None
        # predict with branch of multi-classification: K-class output (dimension=K)
        logits_output_bom = self.head_bom(feature_2)

        return logits_output_bos


    def shap_bom(self, x_1, x_2):
        # x_1 is always ok sample
        # x_2 can be ok or ng sample

        # encoder generate feature embeddings of dimension (H,W,M)
        feature_2 = self.cnn_encoder(x_2)
        # print(feature_2.shape)
        if x_1 is not None:
            feature_1 = self.cnn_encoder(x_1)
            # predict with branch of similarity: binary output (dimension=2)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)
            logits_output_bos = self.head_bos(feature_concat)
        else:
            logits_output_bos = None
        # predict with branch of multi-classification: K-class output (dimension=K)
        logits_output_bom = self.head_bom(feature_2)

        return logits_output_bom
