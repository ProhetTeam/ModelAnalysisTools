import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import copy

try:
    from thirdparty.mtransformer.DSQ.DSQConv import DSQConv
    from thirdparty.mtransformer.APOT.APOTLayers import APOTQuantConv2d
    from thirdparty.mtransformer.LSQ.LSQConv import LSQConv2d
except ImportError:
    raise ImportError("Please import ALL Qunat layer!!!!")

from collections import OrderedDict
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
import numpy as np
from functools import partial

import plotly.express as px
import plotly.offline as of
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.figure_factory as ff
import pandas as pd
of.offline.init_notebook_mode(connected=True)

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class ModelAnalyticalTool:
    r"""This class for analysis model weights and activation distribution """
    """Save forward input and output: Float32 model and Int Model"""
    _version: int = 1

    _layers_input_dict = OrderedDict()
    _layers_output_dict = OrderedDict()
    _layers_bn_dict = OrderedDict()

    def __init__(self, 
                 model: nn.Module,
                 is_quant: bool = False,
                 is_train: bool = False,
                 save_path: str = 'model_analysis.html') -> None:
        """
        Initializes model and plot figure.
        """
        self.model = model
        self.save_path = save_path
        self.is_quant = is_quant

        self.quant_layers = (DSQConv, LSQConv2d, APOTQuantConv2d)
        subplot_titles = ('Weight Distribution',  'Activation Distribution')
        specs = [[{"type": "Histogram"}, {"type": "Histogram"}]]

        if is_quant:
            specs = [specs[0], [{"type": "scatter"}, {"type": "Scatter"}]]
            subplot_titles = subplot_titles + ('Weight Quant Similarity',  'Activation Quant Similarity') 

        self.fig = make_subplots(
            rows = 2 if is_quant else 1, cols = 2,
            column_widths=[0.5, 0.5],
            specs = specs,
            subplot_titles= subplot_titles) 

        if is_train:
            self.model.train()
        else:
            self.model.eval()
        self.init_hooks()

    def init_hooks(self):
        for n, m in self.model.named_children():
            self.register_forward_hook(n, m)
    
    def register_forward_hook(self, name, module):
        for n, m in module.named_children():
            self.register_forward_hook(".".join([name, n]), m)
        if len(list(module.children())) == 0:
            hook_fn_w_name = partial(self.hook_fn, name=name)
            module.register_forward_hook(hook_fn_w_name)
    
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
        if name not in ["conv1", "bn1", "relu", "maxpool"]:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._layers_input_dict[name] = input[0].cpu().detach().numpy().copy()
            elif isinstance(module, nn.ReLU):
                self._layers_output_dict[name] = output.cpu().detach().numpy().copy()
            elif isinstance(module, nn.BatchNorm2d):
                self._layers_bn_dict[name] = output.cpu().detach().numpy().copy()

    def weight_dist_analysis(self, smaple_num = 10, 
                             max_data_length : int = 2e4, 
                             bin_size:float= 0.01):
        r""" Model weigths distribution Statistics """            
        print('Start to analyze Weigths Distribution')
        
        parameters_dict = OrderedDict()
        for name, para in self.model.named_parameters():
            if 'conv' in name and name.endswith('weight') and 'gn' not in name:
                parameters_dict[name] = para.cpu().flatten().detach().numpy()
        
        '''
        interval = int(len(parameters_dict) // smaple_num)
        smaple_para_names = list(parameters_dict)[0:len(parameters_dict): interval][0:smaple_num]
        para_data =  [parameters_dict[para_name] for para_name in smaple_para_names] 
        para_names = [name for name in smaple_para_names] 
        fig_temp = ff.create_distplot(self.down_sample_data(para_data, max_data_length), 
                                        para_names,
                                        histnorm='probability',
                                        bin_size = bin_size, 
                                        show_curve = False,
                                        show_rug = False)
        '''
        if not self.is_quant:
            sample_parameters_dict = self.sampleN_dict(parameters_dict, sample_num = sample_num)
            fig_temp = self.displot_plotly(sample_parameters_dict, max_data_length, bin_size)
            [self.fig.add_trace(go.Histogram(ele), 
                            row = 1, col = 1) for ele in fig_temp['data']]
                        
        self._parameters_dict = parameters_dict
        print('Weigths Distribution Analysis is DONE!')
    
    def displot_plotly(self, data_dict: OrderedDict(),
                       max_data_length: int = 2e4,
                       bin_size:float = 0.01):
        data_list = [v for _, v in data_dict.items()]
        data_name_list = [name for name, _ in data_dict.items()]
        fig = ff.create_distplot(self.down_sample_data(data_list, max_data_length), 
                                        data_name_list,
                                        histnorm='probability',
                                        bin_size = bin_size, 
                                        show_curve = False,
                                        show_rug = False)
        return fig
    
    def sampleN_dict(self, data_dict: OrderedDict(), sample_num = -1):
        if sample_num == -1 or len(data_dict) <= sample_num:
            return data_dict
        smaple_result = OrderedDict()
        interval = int(len(data_dict) // sample_num)
        smaple_data_names = list(data_dict)[0::interval][0:sample_num]

        for name in smaple_data_names:
            smaple_result[name] = data_dict[name]
        return smaple_result

    def draw_quant_weights(self,
                          smaple_num = 10,
                          max_data_length: int = 2e4,
                          bin_size = 0.02):
        sample_quant_weight = self.sampleN_dict(self._quant_weight, smaple_num)
        sample_float_weight = OrderedDict({k[6:]: self._parameters_dict[k[6:] + '.weight'] for k in sample_quant_weight})

        r"""1. Compute Weights Similarity """ 
        sample_para_diff_dict = self.cos_difference_metric(sample_quant_weight, sample_float_weight)
        self.fig.add_trace(
            go.Scatter(
                x = list(sample_para_diff_dict.keys()),
                y = [val for _, val in sample_para_diff_dict.items()],
                mode='markers',
                marker=dict(size=16, color = np.random.randn(len(sample_para_diff_dict)), colorscale='Viridis', 
                showscale = False)
            ),row = 2, col = 1)
        
        r"""2. Compute Activation Similarity"""
        sample_float_weight.update(sample_quant_weight)
        fig_temp = self.displot_plotly(sample_float_weight, max_data_length, bin_size)
        [self.fig.add_trace(go.Histogram(ele), 
                        row = 1, col = 1) for ele in fig_temp['data']] 

    def draw_quant_activation(self,
                              smaple_num = 10,
                              max_data_length: int = 2e4,
                              bin_size = 0.02):
        sample_quant_act = self.sampleN_dict(self._quant_activation, smaple_num)
        sample_float_act = OrderedDict({k[6:]: self._layers_input_dict[k[6:]].flatten() for k in sample_quant_act})

        r"""1. Compute Weights Similarity """
        sample_act_diff_dict = self.cos_difference_metric(sample_quant_act, sample_float_act)
        self.fig.add_trace(
            go.Scatter(
                x = list(sample_act_diff_dict.keys()),
                y = [val for _, val in sample_act_diff_dict.items()],
                mode='markers',
                marker=dict(size=16, color = np.random.randn(len(sample_act_diff_dict)), colorscale='Viridis', 
                showscale = False)
            ),row = 2, col = 2)

        r"""2. Compute Activation Similarity"""
        sample_float_act.update(sample_quant_act)
        fig_temp = self.displot_plotly(sample_float_act, max_data_length, bin_size)
        [self.fig.add_trace(go.Histogram(ele), 
                        row = 1, col = 2) for ele in fig_temp['data']] 
        
    def cos_difference_metric(self, dict_a: OrderedDict(), dict_b: OrderedDict()) -> OrderedDict():
        from numpy import dot
        from numpy.linalg import norm

        res = OrderedDict()
        for key in dict_a:
            a = dict_a[key]
            try:
                b = dict_b[key]
            except:
                b = dict_b[key[6:]]
            res[key] = dot(a, b) / (norm(a) * norm(b)) * 100
        return res

    def extract_quant_info(self):
        self._quant_weight = OrderedDict()
        self._quant_activation = OrderedDict() 
        def get_quant_variable(name, module):
            for n, m in self.model.named_children():
                for n, m in module.named_children():
                    get_quant_variable(".".join([name, n]), m)
                if len(list(module.children())) == 0 \
                        and isinstance(module, self.quant_layers):
                    try:
                        self._quant_weight["quant." + name] = module.Qweight.cpu().detach().numpy().copy().flatten()
                        self._quant_activation["quant." + name] = module.Qactivation.cpu().detach().numpy().copy().flatten()
                    except AttributeError as error:
                        raise error("Your quantization layer should have \
                             self._quant_weight and self._quant_activation member!!!")
        for n, m in self.model.named_children():
            get_quant_variable(n, m) 

    def activation_dist_analysis(self, 
                                 infer_func, 
                                 smaple_num = 10,
                                 max_data_length: int = 2e4, 
                                 bin_size: float = 0.02, 
                                 *args,  
                                 **kwargs):
        print('Start to analyze activation Distribution')
        output_float = infer_func(self.model, *args, **kwargs)

        if self.is_quant:
            self.extract_quant_info()
            self.draw_quant_weights(smaple_num, max_data_length, bin_size)
            self.draw_quant_activation(smaple_num, max_data_length, bin_size)
            print('Activation Distribution Analysis is DONE')
            return

        sample_interval = int(len(self._layers_input_dict) // smaple_num)
        for conv_name in list(self._layers_input_dict)[1::sample_interval][0:smaple_num]: # Skip 1th layer, because its feature map is too big 
            conv_input = self._layers_input_dict[conv_name].flatten()
            bn_name = conv_name[0:-6]+'.bn1'
            bn_output = self._layers_bn_dict[bn_name].flatten() if bn_name in self._layers_bn_dict else np.array([0])
            r"""TODO: relu leky-Relu etc : @Tanfeiyang """ 
            act_name = conv_name[0:-6]+'.relu'
            act_output = self._layers_output_dict[act_name].flatten() if act_name in self._layers_output_dict else np.array([0])

            data = [conv_input, bn_output, act_output]
            fig_temp = ff.create_distplot(self.down_sample_data(data, 2e4), 
                                           [conv_name, bn_name, act_name],
                                           histnorm='probability',
                                           bin_size = bin_size, 
                                           show_curve = False,
                                           show_rug = False)
            [self.fig.add_trace(go.Histogram(ele), 
                            row = 1,
                            col = 2) for ele in fig_temp['data']]

        print('Activation Distribution Analysis is DONE') 
    
    def down_sample_data(self, data: list = [], sample_length:int = -1):
        r""" This function is designed to sample data, but distribution is simillar to orginnal data
        Args:
            data: list[Numpy]
            sample_length: int, how many data u will sample
        return:
            data: list[Numpy]
        """
        if sample_length == - 1:
            return data

        res = []
        for ele in data:
            if len(ele) <= sample_length:
               res.append(ele)
               continue 
            sample_interval = int(len(ele) // sample_length)
            res.append(ele[0::sample_interval][0:int(sample_length)])
            
        return res

    def down_html(self):
        self.fig.update_layout(
            template="plotly_dark",
            xaxis_title='weight_value',
            yaxis_title='ratio'
        )
        self.fig['layout']['xaxis']['title']='Value'
        self.fig['layout']['yaxis']['title']='Ratio'
        self.fig['layout']['xaxis2']['title']='Value'
        self.fig['layout']['yaxis2']['title']='Ratio'

        self.fig.write_html(self.save_path) 