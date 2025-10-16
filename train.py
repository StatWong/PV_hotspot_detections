from ultralytics import YOLO

def main():
    model = YOLO("yolo12l.yaml")
    
    # 训练参数
    results = model.train(
        data="my_data.yaml",          
        epochs=150,                
        imgsz=640,                   
        batch=32,                     
        name="my_custom_training",     # 保存目录名
        device="0",                 
        workers=8,                     # 进程数
        patience=30,                   # 早停
        save=True,            
        amp=True,                      # 自动混合
        single_cls=False,       
        verbose=True,                  # 显示详细训练信息
        cos_lr = True,                  # 余弦退火
        lr0=0.0025,                   
        lrf=0.1,                  
        momentum=0.937,             
        weight_decay=0.0005,           # 权重衰减
        warmup_epochs=3.0,             # 学习率预热轮次
        warmup_momentum=0.8,           # 预热动量
        box=7.5,                       # 边界框损失权重
        cls=0.5,                       # 分类损失权重
        dfl=1.5,                       # 分布焦点损失权重
        hsv_h=0.015,                   # 图像HSV-Hue增强
        hsv_s=0.7,                     # 图像HSV-Saturation增强
        hsv_v=0.4,                     # 图像HSV-Value增强
        degrees=0.0,                   # 图像旋转角度
        translate=0.1,                 # 图像平移
        scale=0.5,                     # 图像缩放
        shear=0.0,                     # 图像剪切
        perspective=0.0,               # 图像透视变换
        flipud=0.0,                    # 上下翻转概率
        fliplr=0.5,                    # 左右翻转概率
        mosaic=0.7,                    # 马赛克数据增强概率
        close_mosaic=10,              # 关闭马赛克增强的轮次
        mixup=0.0,                     # MixUp数据增强概率
        copy_paste=0.0,                # 复制粘贴数据增强概率
        erasing=0.2,                   # 随机擦除概率
        auto_augment="randaugment",  # 自动增强策略
        pretrained=True             # 预训练
    )
    
    print("训练完成！")
    
    # 验证最佳模型
    print("开始验证最佳模型...")
    model.val(
        data="my_data.yaml",
        batch=32,
        imgsz=640,
        conf=0.001,
        iou=0.6,
        device="0"
    )

if __name__ == '__main__':
    main()