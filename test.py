import os
import re
import time
from datetime import datetime
from collections import Counter
from ultralytics import YOLO

# ================================
# 按需改动
# ================================
MODEL_WEIGHTS = 'runs/detect/my_custom_training5/weights/best.pt'  # 模型权重路径
SOURCE_DIR    = 'images/test'                                      # 推理输入目录/视频/图片/通配符
PROJECT_DIR   = 'runs/detect'                                      # 输出父目录
RUN_NAME      = 'final_results'                                    # 本次运行输出子目录
CONF_THRESH   = 0.25                                               # 置信度阈值
IOU_THRESH    = 0.45                                               # NMS IoU阈值
IMG_SIZE      = 640                                                # 推理输入尺寸
DEVICE        = 0                                                  
EXIST_OK      = True                                               # 目录存在时是否复用
SHOW_WINDOW   = False                                              # 是否弹窗可视化（服务器上建议 False）
# ================================
# 以下通常无需修改
# ================================

# 加载模型
model = YOLO(MODEL_WEIGHTS)

infer_kwargs = dict(
    source=SOURCE_DIR,
    save=True,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=EXIST_OK,
    show=SHOW_WINDOW,
    conf=CONF_THRESH,
    iou=IOU_THRESH,
    imgsz=IMG_SIZE,
    device=DEVICE
)

# 推理计时
start_time = time.time()
results = model.predict(**infer_kwargs)
end_time = time.time()

total_time = end_time - start_time
num_images = len(results)
avg_time = (total_time / num_images) if num_images > 0 else 0.0
fps = (num_images / total_time) if total_time > 0 else 0.0

# 平均耗时与类别统计
avg_pre = avg_inf = avg_post = 0.0
from collections import Counter
cls_counter = Counter()

if num_images > 0:
    pre_list, inf_list, post_list = [], [], []
    for r in results:
        sp = getattr(r, "speed", {}) or {}
        if "preprocess" in sp: pre_list.append(sp["preprocess"])
        if "inference"  in sp: inf_list.append(sp["inference"])
        if "postprocess" in sp: post_list.append(sp["postprocess"])

        if getattr(r, "boxes", None) is not None and getattr(r.boxes, "cls", None) is not None:
            for cid in r.boxes.cls.tolist():
                cls_counter[int(cid)] += 1

    if pre_list:  avg_pre  = sum(pre_list)  / len(pre_list)
    if inf_list:  avg_inf  = sum(inf_list)  / len(inf_list)
    if post_list: avg_post = sum(post_list) / len(post_list)

# 类别名称映射
names = getattr(results[0], "names", {}) if num_images > 0 else getattr(model, "names", {})
cls_lines = []
for cid, cnt in cls_counter.most_common():
    cname = names.get(cid, str(cid))
    cls_lines.append(f"- {cname} (id={cid}): {cnt}")

# 输出目录
if num_images > 0 and hasattr(results[0], "save_dir"):
    save_dir = str(results[0].save_dir)
else:
    save_dir = os.path.join(PROJECT_DIR, RUN_NAME)
os.makedirs(save_dir, exist_ok=True)

# 从模型路径提取训练编号（my_custom_trainingXX → XX）
model_path = getattr(model, "ckpt_path", MODEL_WEIGHTS)
match = re.search(r"my_custom_training(\d+)", model_path)
model_id = match.group(1) if match else "0"
summary_path = os.path.join(save_dir, f"summary_{model_id}.txt")

# 报告部分
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("YOLO 推理摘要报告\n")
    f.write("=" * 40 + "\n")
    f.write(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"模型权重：{model_path}\n")
    f.write("\n[推理参数]\n")
    for k, v in infer_kwargs.items():
        f.write(f"- {k}: {v}\n")

    f.write("\n[总体性能]\n")
    f.write(f"- 处理图片数：{num_images}\n")
    f.write(f"- 总耗时：{total_time:.4f} s\n")
    f.write(f"- 平均每张耗时：{avg_time:.6f} s\n")
    f.write(f"- FPS：{fps:.2f}\n")

    f.write("\n[平均分项耗时（每张，毫秒）]\n")
    f.write(f"- preprocess：{avg_pre:.3f} ms\n")
    f.write(f"- inference ：{avg_inf:.3f} ms\n")
    f.write(f"- postprocess：{avg_post:.3f} ms\n")

    f.write("\n[检测到的类别统计]\n")
    f.write("\n".join(cls_lines) + "\n" if cls_lines else "- （无检测或未统计到类别）\n")

print(f"✅ 推理完成，统计信息已写入：{summary_path}")
print(f"📁 可视化结果与 TXT 位于同一文件夹：{save_dir}")


