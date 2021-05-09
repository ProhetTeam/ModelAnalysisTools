## Welcome to use Model analysis tool !

### 1. Tutorial

[1] Single Model analysis command, Model can be float32 or INTx


```
python3 tools/model_analysis.py img-path work_dirs/atss_r50_fpn_1x_coco_Nuscenes_exp4_3cls_QAT_4bit/atss_r50_fpn_1x_coco_Nuscenes_exp4_3cls_QAT_4bit.py  work_dirs/atss_r50_fpn_1x_coco_Nuscenes_exp4_3cls_QAT_4bit/latest.pth --device cpu
```

[2] Two Models analysis command: Float And Quant Model

```
python3 tools/model_analysis_v2.py 
 img-path
 --config-float float-config-path \
 --config-int   int-config-path \
 --checkpoint-float float-model.pth \ 
 --checkpoint-int int-model.pth \
 --device cpu
```
[3] Ananlysis One Model: Weight and Activation Distribution

```
python3 tools/one_model_analysis_tool.py 1260,3f000afab0a06 \
--config thirdparty/configs/LSQDPlus/config10_res18_lsqdplus_2w4f_addoffset_lr4x_4m.py \
--checkpoint work_dirs/LSQDPlus/config10_res18_lsqdplus_2w4f_addoffset_lr5x_4m/latest.pth \
--device cuda
```
