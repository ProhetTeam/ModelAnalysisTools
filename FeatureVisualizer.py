import argparse
import copy
import os, sys
import os.path as osp
import time
import cv2
import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import checkpoint, init_dist
from mmcv.runner import load_checkpoint
from argparse import ArgumentParser, Namespace

from lbitcls import __version__
from lbitcls import models
from lbitcls.datasets import build_dataset
from lbitcls.utils import collect_env, get_root_logger
from lbitcls.models import build_classifier
from lbitcls.apis import set_random_seed, train_classifier
from lbitcls.datasets.pipelines import Resize, CenterCrop

from thirdparty.mtransformer import build_mtransformer
from functools import partial
from thirdparty.model_analysis_tool.MultiModelCmp import MultiModelCmp
from lbitcls.apis import init_model, inference_model

import seaborn as sns
import torch
from torch.optim import Adam
import random
import numpy as np
from pylab import subplot
import matplotlib.pyplot as plt
import pandas as pd

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def infer(model, img):
    out= model(img, return_loss=False)
    return out


from functools import partial
from collections import OrderedDict
import torch.nn as nn
from thirdparty.model_analysis_tool.MultiModelCmp import OneModelDeploy
try:
    from thirdparty.mtransformer.DSQ.DSQConv import DSQConv
    from thirdparty.mtransformer.APOT.APOTLayers import APOTQuantConv2d
    from thirdparty.mtransformer.LSQ.LSQConv import LSQConv2d
    from thirdparty.mtransformer.LSQPlus import LSQDPlusConv2d
except ImportError:
    raise ImportError("Please import ALL Qunat layer!!!!")

QUATN_LAYERS = (DSQConv, LSQConv2d, APOTQuantConv2d, LSQDPlusConv2d)

class OneModelDeployV2(OneModelDeploy):
        
    def hook_fn(self, module, input, output, name, is_quant = False):
        r""" Forward  Hooks 
        Args: 
            model:  nn.model
            input:  Module input tensor
            output: Module output tensor
            name:   Module layer name
        Action:        
            save nn.Conve and nn.Linear Input Tensor
            save BatchNorm and Relu layer output 
        """
        #if name not in ["conv1", "bn1", "relu", "maxpool"]:
        if True:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._layers_input_dict[name] = input[0].cpu().detach()
            elif isinstance(module, nn.ReLU):
                self._layers_output_dict[name] = output.cpu().detach()
            elif isinstance(module, nn.BatchNorm2d):
                self._layers_bn_dict[name] = output.cpu().detach()
    
    def extract_model_info(self):
        for name, module in self.model.named_modules():
            if isinstance(module, QUATN_LAYERS):
                try:
                    self._quant_weight[name] = module.Qweight.cpu().detach()
                    self._quant_activation[name] = module.Qactivation.cpu().detach()
                except:
                    self._quant_weight[name] = None
                    self._quant_activation[name] = None
                    
            if isinstance(module, nn.Conv2d): 
                self._weight[name] = module.weight.cpu().detach()
                
class FeatureVisulizer:
    def __init__(self,
                 models,
                 sample_num = 10,
                 sample_channel = 5,
                 is_train: bool = False,
                 seed = 10,
                 save_path: str = 'features.jpg',
                 extra_names = None,
                 use_torch_plot: bool = True) -> None:
        """
        Initializes model and plot figure.
        """
        extra_names = range(0, len(models)) if extra_names is None else extra_names
        self.models = OrderedDict()
        for name, model in zip(extra_names, models):
            self.models[name] = OneModelDeployV2(model)
        
        self.sample_num = sample_num
        self.sample_channel = sample_channel
        self.seed = seed
        self.save_path = save_path
    
    def __call__(self, infer_func, *args, **kwargs):
        for name, _ in self.models.items():
            self.models[name](infer_func, *args, **kwargs)

        self.feature_visulization()
    
    def feature_visulization(self):
        sample_func1 = partial(self.sampler, sample_num = self.sample_num)
        self.sample_feats = OrderedDict()
        for name, model in self.models.items():
            if hasattr(model, '_quant_activation') and len(model._quant_activation) > 10000:
                temp_res = sample_func1(model._quant_activation)
                print('log')
            else:
                temp_res = sample_func1(model._layers_input_dict)
            self.sample_feats.update(OrderedDict({name + '.' + k: v for k, v in temp_res.items()}))
        
        lbd= lambda t: t[0][len(t[0].split('.')[0]):]
        self.sample_feats = OrderedDict(sorted(self.sample_feats.items(), key=lbd))
        self.draw_features(self.sample_feats)
        
    def draw_features(self, features: OrderedDict):
        self.rows = int(len(features) + 2 * len(features) / len(self.models))
        self.cols = self.sample_channel
        print(self.rows, self.cols)
        
        sns.set(style='darkgrid', rc={"figure.figsize": (6 * self.cols, 6 * self.rows)} )
        
        row_idx = 0
        cache_name = []
        cache_feat = []
        for feat_name, feat in features.items():
            cache_name.append(feat_name)
            random.seed(self.seed)

            feat = self.expand_channel(feat, self.cols - 1)
            feat = torch.cat((feat, torch.sum(feat, 1).unsqueeze(1)), 1)
            cache_feat.append(feat)

            channels_idx = np.sort(random.sample(range(feat.shape[1] - 1), self.cols - 1))
            channels_idx = np.append(channels_idx, feat.shape[1] - 1)
            print(feat.shape, channels_idx)

            r"""1. Draw Different Channel Feature Maps """
            for idx, c in enumerate(channels_idx):
                feature = feat[0,c,...]
                norm_feature = self.convert_rgb(feature)

                subplot(self.rows, self.cols, row_idx * self.cols + idx + 1)
                cbar_ = True if idx == self.cols - 1 else False
                ax = sns.heatmap(data = norm_feature,  cbar=cbar_)
                ax.set_xlabel("channel_{}".format(c))
                if idx == 0:
                    ax.set_ylabel("{}".format(feat_name))
            row_idx +=1
            
            r"""2. Draw Channel distribution """
            if row_idx % (len(self.models) + 2) == len(self.models):
                for idx, c in enumerate(channels_idx):
                    df = self.convert_dataframe(cache_name, cache_feat, c, norm = False, bins = 200)
                    subplot(self.rows, self.cols, row_idx * self.cols + idx + 1)
                    ax = sns.lineplot(data=df, x="Value", y="Ratio", hue="Model_type", markers= '1', dashes=False)
                    ax.set_xlabel("channel_{}".format(c))

                row_idx +=1
            
            r"""3. Draw norm Channel distribution """
            if (row_idx - 1) % (len(self.models) + 2) == len(self.models):
                for idx, c in enumerate(channels_idx):
                    df = self.convert_dataframe(cache_name, cache_feat, c, norm = True, bins = 200)
                    subplot(self.rows, self.cols, row_idx * self.cols + idx + 1)
                    ax = sns.lineplot(data=df, x="Value", y="Ratio", hue="Model_type", markers= '1', dashes=False)
                    ax.set_xlabel("channel_{}".format(c))

                cache_name = []
                cache_feat = []
                row_idx +=1
                
        #plt.show()
        plt.savefig(self.save_path)
        
    def convert_dataframe(self, cache_name, cache_feat, channel_idx, bins = 200, norm = False):
        x_sum = []
        y_sum = []
        model_type = []
        bins = 200
        for n_idx, n in enumerate(cache_name):
            feature = cache_feat[n_idx][0,channel_idx,...]
            if norm:
                feature = (feature -feature.mean())/(feature.std() + 1e-7)

            y_temp = feature.histc(bins = bins)/feature.numel() * 100
            x_temp = torch.linspace(feature.min(), feature.max(), steps = bins).cpu()
            model_type_temp = np.array([n.split('.')[0] for _ in range(bins)])

            x_sum = torch.cat((x_sum, x_temp)) if x_sum != [] else x_temp
            y_sum = torch.cat((y_sum, y_temp)) if y_sum != [] else y_temp
            model_type = np.concatenate((model_type, model_type_temp)) if not isinstance(model_type, list) else model_type_temp

        data = torch.stack((x_sum.t(), y_sum.t()),1).detach().float().cpu().numpy()
        df = pd.DataFrame(data, columns = ['Value','Ratio'])
        df['Model_type'] = model_type
        return df

    def expand_channel(self, data, target_channels):
        if data.shape[1] < target_channels:
            cpad_num = target_channels - data.shape[1]
            data = torch.cat((data, torch.zeros(1, cpad_num, data.shape[-2], data.shape[-1])), 1)
        return data

    def convert_rgb(self, data:torch.tensor):
        data = data.clone()
        res = (data - data.mean())/data.std() 
        res = (res * 64 + 128).int().clamp(0, 255)
        return res

    def sampler(self, data, sample_num = -1):
        if isinstance(data, OrderedDict):
            if sample_num == -1 or len(data) <= sample_num:
                return data
            interval = int(len(data) // sample_num)
            smaple_result = OrderedDict()
            smaple_data_names = list(data)[0::interval][0:sample_num]
            for name in smaple_data_names:
                smaple_result[name] = data[name]
            return smaple_result
        elif isinstance(data, list): #data: list[Numpy]
            if sample_num == - 1:
                return data
            smaple_result = []
            res = []
            for ele in data:
                if len(ele) <= sample_num:
                    res.append(ele)
                    continue 
                sample_interval = int(len(ele) // sample_num)
                res.append(ele[0::sample_interval][0:int(sample_num)])
            return res

img = '1260,3f000afab0a06'
#img = '1260,1c0004e302a9f'
#img = '1260,1000fe3516a5'

'''
configs = ['thirdparty/configs/benchmark/config1_res18_float_1m_b64.py',
           'thirdparty/configs/LSQDPlus/config4_res18_lsqdplus_int4_updatelr4x_weightloss_4m.py',
           'thirdparty/configs/LSQDPlus/config6_res18_lsqdplus_int3_allchangenoweightloss_4m.py',]
checkpoints = ['thirdparty/modelzoo/res18.pth',
               'work_dirs/LSQDPlus/config4_res18_lsqdplus_int4_updatelr4x_weightloss_4m/latest.pth',
               'work_dirs/LSQDPlus/config6_res18_lsqdplus_int3_allchangenoweightloss_4m/latest.pth']
extra_names = ['r18-fp32', 
               'r18-int4',
               'r18-int3']
'''

'''
configs = ['thirdparty/configs/benchmark/config7_mobilev2_float_2m_b64_coslr.py',
           'thirdparty/configs/LSQDPlusBack/config20_mobilenetv2_lsqdplus_int4_addoffset_lr4x_selfback_4m.py',
           'thirdparty/configs/LSQDPlus/config13_mobilenetv2_lsqdplus_int3_addoffset_lr4x_4m.py',
           'thirdparty/configs/LSQDPlus/config14_mobilenetv2_lsqdplus_2w4f_addoffset_lr4x_4m.py',
           ]
checkpoints = ['thirdparty/modelzoo/MobileNetV2.pth',
               'work_dirs/LSQDPlusBack/config20_mobilenetv2_lsqdplus_int4_addoffset_lr4x_selfback_4m_new/latest.pth',
               'work_dirs/LSQDPlus/config13_mobilenetv2_lsqdplus_int3_addoffset_lr4x_4m/latest.pth',
               'work_dirs/LSQDPlus/config14_mobilenetv2_lsqdplus_2w4f_addoffset_lr4x_4m/latest.pth']
extra_names = ['MBV2-fp32', 
               'MBV2-int4',
               'MBV2-int3',
               'MBV2-2w4f']

'''

configs = ['thirdparty/configs/benchmark/config5_res50_float_2m_b64_coslr.py',
           'thirdparty/configs/LSQDPlus/config9_res50_lsqdplus_int4_addoffset_lr4x__4m.py',
           'thirdparty/configs/LSQDPlus/config10_res50_lsqdplus_int3_addoffset_coslr4x__4m.py',
           'thirdparty/configs/LSQDPlus/config11_res50_lsqdplus_2w4f_addoffset_lr4x__4m.py',
           ]
checkpoints = ['thirdparty/modelzoo/res50.pth',
               'work_dirs/LSQDPlus/config9_res50_lsqdplus_int3_addoffset_lr4x__4m/epoch_108.pth',
               'work_dirs/LSQDPlus/config10_res50_lsqdplus_int3_addoffset_coslr4x__4m/latest.pth',
               'work_dirs/LSQDPlus/config11_res50_lsqdplus_2w4f_addoffset_lr4x__4m/latest.pth']
extra_names = ['Res50-fp32', 
               'Res50-int4',
               'Res50-int3',
               'Res50-2w4f']


save_path = 'features.jpg'
device = 'cpu'

models = []
for idx, config in enumerate(configs):
    models.append(init_model(config, checkpoints[idx], device = device))

feat_visulizer = FeatureVisulizer(models, 
                                  extra_names = extra_names, 
                                  sample_num = 8,
                                  sample_channel = 8, 
                                  save_path= save_path)
feat_visulizer(inference_model, img = img)