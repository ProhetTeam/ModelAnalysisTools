from argparse import ArgumentParser

import sys
sys.path.append('../')

from analysistools.OneModelAnalyticalTool import OneModelAnalysis
from lowbit_classification.lbitcls.apis import init_model, inference_model

def infer(model, img):
    out= model(img, return_loss=False)
    return out

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default='1260,1180006216d1c0')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help = 'Int checkpoint file')
    parser.add_argument('--save-path', type = str, default= "./model_analysis.html", help = "html save path")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--quant_layers', type=list, help='which quant layers you want to analysis', default=('LSQConv2d', 'DSQConv2d'))
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device = args.device)
    r""" 1. Float32 Model analysis """
    model_analysis_tool = OneModelAnalysis(model, 
                                         smaple_num = 20, 
                                         max_data_length = 2e4,
                                         quant_layers=args.quant_layers, 
                                         bin_size = 0.01, 
                                         save_path = args.save_path,
                                         use_torch_plot = True)
    model_analysis_tool(inference_model, img = args.img)
    model_analysis_tool.down_html()
    
if __name__ == '__main__':
    main()
