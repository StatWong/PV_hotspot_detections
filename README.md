# PV_hotspot_detections

## 建议目录结构
```
V12/
├─ .vscode/
│   └─ settings.json
├─ images/                 # 原始图像（按数据划分）
│   ├─ train/
│   ├─ val/
│   └─ test/
├─ labels/                 # YOLO标注txt（与images一一对应）
│   ├─ train/
│   ├─ val/
│   └─ test/
│   ├─ train.cache         # 可选：数据索引缓存
│   └─ val.cache
├─ runs/                   # 训练/验证/推理输出（自动生成）
│   └─ detect/
│       ├─ my_custom_training/
│       ├─ my_custom_training2/
│       ├─ my_custom_training3/
│       ├─ my_custom_training4/
│       ├─ my_custom_training5/ #目前最好
│       ├─ my_custom_training6/ #yolov12，未训练完成，可接续
│       ├─ val/
│       ├─ val2/
│       ├─ val3/
│       └─ final_results/
├─ bus.jpg
├─ check_yaml.py
├─ class_count.py
├─ my_data.yaml
├─ test.py
└─ train.py
```


## 快速使用

### 环境
requriements.txt

## 脚本说明
- **check_yaml.py**：检查yaml可用性
- **class_count.py**：统计类别数量与样本分布
- **train.py**：训练脚本
- **test.py**：推理脚本

## 结果管理
每次训练会自动在 `runs/detect/` 下生成独立文件夹，方便A/B测试与结果复现。
