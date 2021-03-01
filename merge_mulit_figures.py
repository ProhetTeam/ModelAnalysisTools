import plotly.io as pio
from collections import OrderedDict
import plotly.graph_objs as go

figures_path = ['model_analysis.json']

scatter_dict = OrderedDict()
for fig_path in figures_path:
    with open(fig_path, 'r') as f:
        fig = pio.from_json(f.read())
        scatter_dict[fig_path]= [ele for ele in fig['data'] \
            if isinstance(ele, go._scatter.Scatter)]

pass