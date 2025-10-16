"""
按后缀分类无人机照片：
- 文件名含 "_T" → 移动到 {日期}热感
- 文件名含 "_V" → 移动到 {日期}自然光
- 不能识别类型或非 JPG/JPEG → 复制到两处（热感/自然光）
- 日期优先取文件名 DJI_YYYYMMDDHHMMSS_*，否则向上找父目录；失败为 "unknown"
- 已在目标目录(…热感/…自然光)内的文件跳过
- DRY_RUN=True 只打印不改动
"""

from __future__ import annotations
import re
import shutil
from pathlib import Path
from typing import Optional

# ==== 修改这里 ====
REAL_DIR = Path(r"C:\jyu\1\real\DJI_202510101513_006") #大疆拍摄的照片所在目录
DRY_RUN = False
# =================

JPG_EXTS = {".jpg", ".jpeg"}
THERMAL_SUFFIX = "热感"
VISIBLE_SUFFIX = "自然光"
UNKNOWN = "unknown"

RE_DJI_DATE = re.compile(r"DJI_(\d{8})\d{6}", re.IGNORECASE)
RE_ANY_8 = re.compile(r"(\d{8})")
RE_T = re.compile(r"(^|_)T(\.|$)", re.IGNORECASE)
RE_V = re.compile(r"(^|_)V(\.|$)", re.IGNORECASE)


def parse_date_from(text: str) -> Optional[str]:
    m = RE_DJI_DATE.search(text)
    if m:
        return m.group(1)
    m2 = RE_ANY_8.search(text)
    return m2.group(1) if m2 else None


def guess_date(p: Path) -> str:
    d = parse_date_from(p.name)
    if d:
        return d
    for parent in p.parents:
        d = parse_date_from(parent.name)
        if d:
            return d
    return UNKNOWN


def detect_type(p: Path) -> Optional[str]:
    s = p.stem
    if RE_T.search(s):
        return "T"
    if RE_V.search(s):
        return "V"
    return None


def ensure_unique(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, suf, parent = dst.stem, dst.suffix, dst.parent
    i = 1
    while True:
        cand = parent / f"{stem}_copy{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def in_target_dir(p: Path) -> bool:
    parent = p.parent.name
    return parent.endswith(THERMAL_SUFFIX) or parent.endswith(VISIBLE_SUFFIX)


def plan_actions(p: Path, root: Path):
    """返回[(op, src, dst), ...]，op 为 'move' 或 'copy'。"""
    date_str = guess_date(p)
    ftype = detect_type(p)
    ext = p.suffix.lower()

    # 非 JPG 或未知类型：复制到两处
    if ext not in JPG_EXTS or ftype is None:
        t1 = root / f"{date_str}{THERMAL_SUFFIX}"
        t2 = root / f"{date_str}{VISIBLE_SUFFIX}"
        t1.mkdir(parents=True, exist_ok=True)
        t2.mkdir(parents=True, exist_ok=True)
        return [
            ("copy", p, ensure_unique(t1 / p.name)),
            ("copy", p, ensure_unique(t2 / p.name)),
        ]

    # JPG 类型移动
    target = root / (
        f"{date_str}{THERMAL_SUFFIX}" if ftype == "T" else f"{date_str}{VISIBLE_SUFFIX}"
    )
    target.mkdir(parents=True, exist_ok=True)
    return [("move", p, ensure_unique(target / p.name))]


def main():
    if not REAL_DIR.exists():
        print(f"错误：目录不存在：{REAL_DIR}")
        return

    print(f"开始分类：{REAL_DIR}")
    total = moved = copied = skipped = 0

    for p in REAL_DIR.rglob("*"):
        if not p.is_file():
            continue
        total += 1
        if in_target_dir(p):
            skipped += 1
            print(f"跳过（已在目标目录）：{p}")
            continue

        actions = plan_actions(p, REAL_DIR)
        for op, src, dst in actions:
            print(f"{'DRY_RUN ' if DRY_RUN else ''}{op.upper():4} {src} -> {dst}")
            if DRY_RUN:
                continue
            if op == "move":
                shutil.move(str(src), str(dst))
                moved += 1
            else:
                shutil.copy2(str(src), str(dst))
                copied += 1

    print("==== 汇总 ====")
    print(f"扫描：{total}")
    print(f"移动：{moved}")
    print(f"复制：{copied}")
    print(f"跳过：{skipped}")
    if DRY_RUN:
        print("（演练模式）")


if __name__ == "__main__":
    main()

