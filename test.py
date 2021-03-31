import torch
import plotly.express as px
import plotly.offline as of
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.figure_factory as ff
import pandas as pd
import torch.nn.functional as F
from collections import OrderedDict
import random
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


config_float = "work_dirs/temp/atss_r18_coco_lsq_float/atss_r18_fpn_1x_coco.py"
checkpoint_float = "work_dirs/temp/atss_r18_coco_lsq_float/atss_r18_coco_lsq_fp.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_float = init_detector(config_float, checkpoint_float, device= device)

weights_dict = OrderedDict()
for name, val in model_float.named_parameters():
    if 'conv' in name and name.endswith('weight'):
        weights_dict[str(name)] = val.cpu().detach().numpy().flatten().copy()

def sampler(data, sample_num :int = -1):
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

weights_dict = sampler(weights_dict, 20)
def plot_dict_torch_plotly(data: OrderedDict(), bins: int = 1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig = go.Figure()
    color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for name, val in data.items():

        val_torch = torch.from_numpy(val)
        """ 1. Generate Y """
        val_temp = val_torch.to(device).histc(bins = bins) / val_torch.shape[0]
        data_list_y = val_temp.cpu().detach().numpy().flatten().copy()
        
        """ 2. Generate x """
        x_temp = torch.linspace(val_torch.min(), val_torch.max(), steps = bins).to(device)
        data_list_x = x_temp.cpu().detach().numpy().flatten().copy()

        """ 3. Draw plotly """
        fig.add_trace(go.Scatter(x = data_list_x, 
                                 y = data_list_y,
                                 name = name,
                                 line=dict( color = color),
                                 mode='lines'))
        pass
    
    fig.write_html('test.html')



plot_dict_torch_plotly(weights_dict, 2000)

max_data_length = 40000 
fig = ff.create_distplot( sampler([val for _, val in weights_dict.items()], max_data_length), 
                            [name for name, _ in weights_dict.items()],
                            histnorm='probability',
                            bin_size = 0.0001,
                            show_curve = False,
                            show_rug = False)
fig.write_html('test.html')
pass