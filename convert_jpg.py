import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

"""  处理内容： 
1. 按 EXIF 自动校正方向 
2. 等比例缩放并填充到 640×640 
3. 灰度化 + 归一化 
4. 随机数据增强（翻转、旋转、亮度对比度调整、加噪、模糊） 

使用方法： 直接修改下面的 input_dir 和 output_dir 路径，然后运行本脚本。 
"""
# ========= 修改这里 =========
INPUT_DIR = r"C:\jyu\1\real\DJI_202510101513_006\20251010热感"
OUTPUT_DIR = r"C:\jyu\1\real\DJI_202510101513_006\20251010热感_processed"
AUGMENT_N = 3  # 每张增强份数（0=不增强）
TARGET_SIZE = 640
# ==========================


def is_jpg(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg"}


def exif_correct(im: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(im)
    except Exception:
        return im


def letterbox(img: np.ndarray, size: int = TARGET_SIZE, color: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def to_gray_norm(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if g.max() > g.min():
        return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return np.zeros_like(g, dtype=np.uint8)


def save_jpg(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        print(f"[错误] JPEG 编码失败：{path}")
        return False
    try:
        buf.tofile(str(path))  # 兼容中文路径
        return True
    except Exception as e:
        print(f"[错误] 保存失败：{path} ({e})")
        return False


# ---------- 数据增强 ----------

def rand_aug(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    if random.random() < 0.5:  # 翻转
        out = cv2.flip(out, 1)
    if random.random() < 0.7:  # 旋转
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-10, 10), 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < 0.7:  # 亮度/对比度
        out = cv2.convertScaleAbs(out, alpha=random.uniform(0.9, 1.1), beta=random.randint(-10, 10))
    if random.random() < 0.5:  # 高斯噪声
        noise = np.random.normal(0, random.uniform(2, 8), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.4:  # 高斯模糊
        k = random.choice([3, 5])
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), k * 100)
    return out


# ---------- 主流程 ----------

def process_one(in_path: Path, out_dir: Path) -> None:
    try:
        with Image.open(in_path) as im:
            im = exif_correct(im).convert("RGB")
            img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[警告] 读取失败，跳过：{in_path} ({e})")
        return

    img = letterbox(img, TARGET_SIZE, 0)
    img = to_gray_norm(img)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = in_path.stem

    if not save_jpg(out_dir / f"{base}_proc.jpg", img):
        return

    for i in range(AUGMENT_N):
        if not save_jpg(out_dir / f"{base}_aug{i + 1}.jpg", rand_aug(img)):
            print(f"[错误] 增强保存失败：{base}_aug{i + 1}.jpg")


def main() -> None:
    src = Path(INPUT_DIR)
    dst = Path(OUTPUT_DIR)
    if not src.exists():
        print(f"[错误] 输入路径不存在：{src}")
        return

    n = 0
    for f in src.rglob("*"):
        if f.is_file() and is_jpg(f):
            process_one(f, dst)
            n += 1
            if n % 20 == 0:
                print(f"[进度] 已处理 {n} 张...")
        elif f.is_file():
            print(f"[跳过] 非 JPG：{f.name}")

    print(f"[完成] 共处理 {n} 张，输出：{dst}")


if __name__ == "__main__":
    main()

