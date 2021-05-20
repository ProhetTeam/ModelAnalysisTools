import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import copy
import inspect
import scipy
from torch.nn import modules
try:
    from thirdparty.mtransformer.DSQ.DSQConv import DSQConv
    from thirdparty.mtransformer.APOT.APOTLayers import APOTQuantConv2d
    from thirdparty.mtransformer.LSQ.LSQConv import LSQConv2d
    from thirdparty.mtransformer.LSQPlus import LSQDPlusConv2d
except ImportError:
    raise ImportError("Please import ALL Qunat layer!!!!")

from collections import OrderedDict
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
import numpy as np
from numpy.linalg import norm
from functools import partial

import plotly.express as px
import plotly.offline as of
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.figure_factory as ff
import pandas as pd
import torch.nn.functional as F

of.offline.init_notebook_mode(connected=True)

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

QUATN_LAYERS = (DSQConv, LSQConv2d, APOTQuantConv2d, LSQDPlusConv2d)

class OneModelDeploy:
    _version: int = 1

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.init_hooks()

        self._layers_input_dict = OrderedDict()
        self._layers_output_dict = OrderedDict()
        self._layers_bn_dict = OrderedDict()
        
        self._quant_weight = OrderedDict()
        self._quant_activation = OrderedDict()
        self._weight = OrderedDict()
    
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
        #if name not in ["conv1", "bn1", "relu", "maxpool"]:
        if True:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._layers_input_dict[name] = input[0].cpu().detach().numpy().flatten().copy()
            elif isinstance(module, nn.ReLU):
                self._layers_output_dict[name] = output.cpu().detach().numpy().flatten().copy()
            elif isinstance(module, nn.BatchNorm2d):
                self._layers_bn_dict[name] = output.cpu().detach().numpy().flatten().copy()

    def __call__(self, infer_func, *args, **kwargs):
        output = infer_func(self.model, *args, **kwargs)
        self.extract_model_info()
    
    def extract_model_info(self):
        for name, module in self.model.named_modules():
            if isinstance(module, QUATN_LAYERS):
                try:
                    self._quant_weight[name] = module.Qweight.cpu().detach().numpy().copy().flatten()
                    self._quant_activation[name] = module.Qactivation.cpu().detach().numpy().copy().flatten()
                except:
                    self._quant_weight[name] = None
                    self._quant_activation[name] = None
                    
            if isinstance(module, nn.Conv2d): 
                self._weight[name] = module.weight.cpu().flatten().detach().numpy()

class MultiModelCmp:
    def __init__(self,
                 models,
                 smaple_num = 10,
                 max_data_length: int = 2e4, 
                 bin_size: float = 0.02, 
                 is_train: bool = False,
                 save_path: str = 'model_analysis.html',
                 extra_names = None,
                 use_torch_plot: bool = True) -> None:
        """
        Initializes model and plot figure.
        """
        extra_names = range(0, len(models)) if extra_names is None else extra_names
        self.models = OrderedDict()
        for name, model in zip(extra_names, models):
            self.models[name] = OneModelDeploy(model)
        self.save_path   = save_path

        """
        Figure Configuration
        """
        self.sample_num = smaple_num
        self.max_data_length = max_data_length
        self.bin_size = bin_size
        self.use_torch_plot = use_torch_plot

        subplot_titles = ('Weight Distribution', 'Activation Distribution')
        specs = [[{"type": "Histogram"}, {"type": "Histogram"}]]
        self.fig = make_subplots(
            rows = 1, cols = 2,
            column_widths=[0.5, 0.5],
            specs = specs,
            subplot_titles= subplot_titles)

    def __call__(self, infer_func, *args, **kwargs):
        for name, _ in self.models.items():
            self.models[name](infer_func, *args, **kwargs)

        self.weight_analysis()
        self.activation_analysis()
        self.down_html()

    def weight_analysis(self):
        sample_func1 = partial(self.sampler, sample_num = self.sample_num)

        sample_weights = OrderedDict()
        for name, model in self.models.items():
            temp_res = sample_func1(model._weight)
            sample_weights.update(OrderedDict({name + '.wgh.' + k: v for k, v in temp_res.items()}))

        self.plot_dict_torch_plotly(sample_weights, row = 1, col = 1)
        

    def activation_analysis(self):
        sample_func1 = partial(self.sampler, sample_num = self.sample_num)

        sample_act = OrderedDict()
        for name, model in self.models.items():
            temp_res = sample_func1(model._layers_input_dict)
            sample_act.update(OrderedDict({name + '.act.' + k: v for k, v in temp_res.items()}))

        self.plot_dict_torch_plotly(sample_act, row = 1, col = 2, mode = "lines")
    
    def displot_plotly(self, data_dict: OrderedDict(),
                       max_data_length: int = 2e4,
                       bin_size:float = 0.01):
        data_list = [v for _, v in data_dict.items()]
        data_name_list = [name for name, _ in data_dict.items()]
        fig = ff.create_distplot(self.sampler(data_list, self.max_data_length), 
                                        data_name_list,
                                        histnorm='probability',
                                        bin_size = bin_size,
                                        show_curve = False,
                                        show_rug = False)
        return fig
    
    def plot_dict_torch_plotly(self,data: OrderedDict(), bins: int = 1000, 
                                row : int = 1, col: int = 1,
                                mode = 'lines'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name, val in data.items():
            val_torch = torch.from_numpy(val)
            """ 1. Generate Y """
            val_temp = val_torch.to(device).histc(bins = bins) / val_torch.shape[0]
            data_list_y = val_temp.cpu().detach().numpy().flatten().copy()
            
            """ 2. Generate x """
            x_temp = torch.linspace(val_torch.min(), val_torch.max(), steps = bins).to(device)
            data_list_x = x_temp.cpu().detach().numpy().flatten().copy()

            """ 3. Draw plotly """
            random.seed(name)
            color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.fig.add_trace(go.Scatter(x = data_list_x, 
                                    y = data_list_y,
                                    name = name,
                                    line = dict(color = color),
                                    mode=mode), row = row, col = col)

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

        self.fig.write_json(self.save_path.split('.html')[0] + '.json')
        self.fig.write_html(self.save_path) 