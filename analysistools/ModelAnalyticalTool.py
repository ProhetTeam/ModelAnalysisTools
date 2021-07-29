from re import S
import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import inspect



from collections import OrderedDict
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

# of.offline.init_notebook_mode(connected=True)

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class ModelDeploy:

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
            if str(type(module)) in QUANT_LAYERS:
                try:
                    self._quant_weight[name] = module.Qweight.cpu().detach().numpy().copy().flatten()
                    self._quant_activation[name] = module.Qactivation.cpu().detach().numpy().copy().flatten()
                except AttributeError:
                    raise AttributeError('Please Trun On Debug!!!!')
            if isinstance(module, nn.Conv2d): 
                self._weight[name] = module.weight.cpu().flatten().detach().numpy()

class DistanceMetric:
    def cos_similarity_metric(self, dict_a: OrderedDict(), dict_b: OrderedDict()) -> OrderedDict():
        res = OrderedDict()
        for key in dict_a:
            a = dict_a[key]
            try:
                b = dict_b[key]
            except:
                b = dict_b[key[6:]]
            res[key] = np.dot(a, b) / (norm(a) * norm(b))
        return res
    
    def KL_divergence(self, dict_p: OrderedDict(), dict_q: OrderedDict(), num_interval: int = 150) -> OrderedDict():
        self.num_interval = num_interval
        res = OrderedDict()

        device = "cuda" if torch.cuda.is_available() else "cpu" 
        for name, val in dict_p.items():
            p = torch.from_numpy(val).to(device)
            q = torch.from_numpy(dict_q[name]).to(device)

            min_num = torch.min(p.min(), q.min())
            max_mun = torch.max(p.max(), q.max())
            p_dist = p.histc(bins = 2000, min = min_num, max = max_mun) / p.shape[0]
            q_dist = q.histc(bins = 2000, min = min_num, max = max_mun) / q.shape[0]
            p_dist[p_dist == 0] = torch.tensor(1e-7, dtype = p_dist.dtype, device = p_dist.device)
            q_dist[q_dist == 0] = torch.tensor(1e-7, dtype = q_dist.dtype, device = q_dist.device)

            r""""
            Other Method: res[name] = F.kl_div(q_dist.log(), p_dist, reduction = 'mean').cpu().detach().numpy()
            """
            res[name] = (p_dist * torch.log2(p_dist/q_dist)).sum().cpu().detach().numpy()

        return res

    def correlation(self, dict_a: OrderedDict(), dict_b: OrderedDict()) -> OrderedDict():
        res = OrderedDict()
        for key in dict_a:
            a = dict_a[key]
            try:
                b = dict_b[key]
            except:
                b = dict_b[key[6:]]
            res[key] = ((a-np.mean(a))*(b-np.mean(b))).mean() / (np.std(a) * np.std(b))
        return res

class QModelAnalysis:
    
    def __init__(self,
                 model_float: nn.Module,
                 model_quant: nn.Module,
                 quant_layers: list,
                 smaple_num = 10,
                 max_data_length: int = 2e4, 
                 bin_size: float = 0.02, 
                 is_train: bool = False,
                 save_path: str = 'model_analysis.html',
                 use_torch_plot: bool = True) -> None:
        """
        Initializes model and plot figure.
        """
        global QUANT_LAYERS
        QUANT_LAYERS = quant_layers
        self.model_float = ModelDeploy(model_float)
        self.model_quant   = ModelDeploy(model_quant)
        self.save_path   = save_path

        """
        Figure Configuration
        """
        self.sample_num = smaple_num
        self.max_data_length = max_data_length
        self.bin_size = bin_size
        self.use_torch_plot = use_torch_plot

        subplot_titles = ('Weight Distribution', 'Activation Distribution', \
                          'Weight Quant Similarity', 'Activation Quant Similarity')
        specs = [[{"type": "Histogram"}, {"type": "Histogram"}],
                 [{"type": "scatter"}, {"type": "Scatter"}]]
        self.fig = make_subplots(
            rows = 2, cols = 2,
            column_widths=[0.5, 0.5],
            specs = specs,
            subplot_titles= subplot_titles)

        self.distance_metrix = DistanceMetric()

    def __call__(self, infer_func, *args, **kwargs):
        self.model_float(infer_func, *args, **kwargs)
        self.model_quant(infer_func, *args, **kwargs)

        self.weight_analysis()
        self.activation_analysis()
        self.down_html()

    def weight_analysis(self):
        sample_func1 = partial(self.sampler, sample_num = self.sample_num)

        assert(len(self.model_float._weight) == len(self.model_quant._weight))
        sample_weight_float = sample_func1(self.model_float._weight)
        sample_weight_fake = sample_func1(self.model_quant._weight)
        sample_weight_quant = sample_func1(self.model_quant._quant_weight)

        r""" 1. Compute weights similarity """
        dis_funcs = inspect.getmembers(DistanceMetric, lambda a: inspect.isfunction(a))

        for dis_func in dis_funcs:
            sample_para_diff1 = dis_func[1](self.distance_metrix, sample_weight_float, sample_weight_fake)
            sample_para_diff2 = dis_func[1](self.distance_metrix, sample_weight_fake, sample_weight_quant)
            self.fig.add_trace(
                go.Scatter(
                    x = list(sample_para_diff1.keys()),
                    y = [val for _, val in sample_para_diff1.items()],
                    name = 'Float.Fake.{}'.format(dis_func[0]),
                    mode = 'lines+markers'
                ),row = 2, col = 1)
            self.fig.add_trace(
                go.Scatter(
                    x = list(sample_para_diff2.keys()),
                    y = [val for _, val in sample_para_diff2.items()],
                    name = 'Fake.Quant.{}'.format(dis_func[0]),
                    mode = 'lines+markers'
                ),row = 2, col = 1)
        
        r""" 2. Weights distribution plot  """
        keys = list(sample_weight_float.keys())
        sample_weight_float.update(OrderedDict({
           'quant.' + k: sample_weight_quant[k] for k in keys}))
        sample_weight_float.update(OrderedDict({
           'fake.' + k: sample_weight_fake[k] for k in keys}))
        
        if self.use_torch_plot:
            self.plot_dict_torch_plotly(sample_weight_float, row = 1, col = 1)
        else:
            fig_temp = self.displot_plotly(sample_weight_float, self.max_data_length, self.bin_size)

            for idx, ele in enumerate(fig_temp['data']):
                ele['marker']['color'] = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            [self.fig.add_trace(go.Histogram(ele), 
                            row = 1, col = 1) for ele in fig_temp['data']]

    def activation_analysis(self):
        sample_func1 = partial(self.sampler, sample_num = self.sample_num)

        sample_act_quant = sample_func1(self.model_quant._quant_activation)
        sample_act_fake = OrderedDict({k:self.model_quant._layers_input_dict[k] for k in sample_act_quant})
        sample_act_float = OrderedDict({k:self.model_float._layers_input_dict[k] for k in sample_act_quant})

        r""" 1. Compute activation similarity """
        dis_funcs = inspect.getmembers(DistanceMetric, lambda a: inspect.isfunction(a))
        for dis_func in dis_funcs: 
            sample_para_diff1 = dis_func[1](self.distance_metrix, sample_act_float, sample_act_fake)
            sample_para_diff2 = dis_func[1](self.distance_metrix, sample_act_fake, sample_act_quant)
            self.fig.add_trace(
                go.Scatter(
                    x = list(sample_para_diff1.keys()),
                    y = [val for _, val in sample_para_diff1.items()],
                    name = 'Float.Fake.{}'.format(dis_func[0]),
                    mode = 'lines+markers'
                ),row = 2, col = 2)
            self.fig.add_trace(
                go.Scatter(
                    x = list(sample_para_diff2.keys()),
                    y = [val for _, val in sample_para_diff2.items()],
                    name = 'Fake.Quant.{}'.format(dis_func[0]),
                    mode = 'lines+markers'
                ),row = 2, col = 2)
        
        r""" 2. Activation distribution plot  """
        keys = list(sample_act_float.keys())
        sample_act_float.update(OrderedDict({
           'quant.' + k: sample_act_quant[k] for k in keys}))
        sample_act_float.update(OrderedDict({
           'fake.' + k: sample_act_fake[k] for k in keys}))
        
        if self.use_torch_plot:
            self.plot_dict_torch_plotly(sample_act_float, row = 1, col = 2)
        else:
            fig_temp = self.displot_plotly(sample_act_float, self.max_data_length, self.bin_size)
            
            for idx, ele in enumerate(fig_temp['data']):
                ele['marker']['color'] = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            [self.fig.add_trace(go.Histogram(ele), 
                            row = 1, col = 2) for ele in fig_temp['data']]
    
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
    
    def plot_dict_torch_plotly(self,data: OrderedDict(), bins: int = 1000, row : int = 1, col: int = 1):
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
            color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.fig.add_trace(go.Scatter(x = data_list_x, 
                                    y = data_list_y,
                                    name = name,
                                    line = dict(color = color),
                                    mode='lines'), row = row, col = col)

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
        