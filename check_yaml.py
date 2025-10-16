# ================================
# 可修改区域（按需改动）
# ================================
YAML_PATH = "my_data.yaml"    # 要检查的 YAML 文件路径
ENCODING = "utf-8"           
# ================================
# 以下通常无需修改
# ================================
import os
import yaml

# 检查文件是否存在
exists = os.path.exists(YAML_PATH)
print(f"YAML 文件存在: {exists}")

if exists:
    # 显示文件内容
    with open(YAML_PATH, "r", encoding=ENCODING) as f:
        content = f.read()
    print("\n=== YAML 文件内容 ===")
    print(content)

    # 尝试解析
    try:
        with open(YAML_PATH, "r", encoding=ENCODING) as f:
            data = yaml.safe_load(f)
        print("\n=== 解析结果 ===")
        print(data)
        print(f"\ntrain 键存在: {'train' in data}")
        print(f"val 键存在: {'val' in data}")
    except Exception as e:
        print(f"\n解析错误: {e}")
else:
    print(f"❌ 未找到文件：{YAML_PATH}")
