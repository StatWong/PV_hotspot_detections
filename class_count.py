import os
import glob
from collections import Counter

# ================================
# 建议修改区域（按需改动）
# ================================
LABELS_FOLDER = "labels/val"   # YOLO 标签文件夹路径
ENCODING = "utf-8"             # 文本编码
# ================================
# 以下通常无需修改
# ================================

def analyze_classes(labels_folder):
    class_counter = Counter()
    total_objects = 0
    
    for txt_file in glob.glob(os.path.join(labels_folder, "*.txt")):
        try:
            with open(txt_file, "r", encoding=ENCODING) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
                        total_objects += 1
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")
    
    return class_counter, total_objects

# 主逻辑
if __name__ == "__main__":
    class_counter, total_objects = analyze_classes(LABELS_FOLDER)

    print("=== 类别统计 ===")
    print(f"总目标对象数: {total_objects}")
    if class_counter:
        print(f"发现的类别ID: {sorted(class_counter.keys())}")
        print(f"类别范围: {min(class_counter.keys())} - {max(class_counter.keys())}")
        print(f"类别数量: {len(class_counter)}")

        print("\n=== 每个类别的实例数量 ===")
        for cid in sorted(class_counter.keys()):
            cnt = class_counter[cid]
            pct = (cnt / total_objects) * 100 if total_objects > 0 else 0
            print(f"类别 {cid}: {cnt} 个实例 ({pct:.2f}%)")
    else:
        print("未在标签文件中检测到任何类别。")
