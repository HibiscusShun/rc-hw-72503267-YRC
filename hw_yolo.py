# -*- coding: utf-8 -*-
"""
T08 YOLO 推理 —— 作业骨架（学生版）

目标：学会加载模型、跑推理、把结果解析成"能用的数据"。
训练是 M08 的事，这里只用官方预训练权重 yolov8n.pt（80 类通用目标）。

自检（本教案不含自动校验脚本）：跑一遍 assets/bus.jpg，看能不能检出 person 与 bus
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- 1
def load_model(weight_path: str):
    """加载 YOLO 模型并返回模型对象。

    from ultralytics import YOLO
    return YOLO(weight_path)
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    from ultralytics import YOLO
    return YOLO(weight_path)
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 2
def detect_objects(model, img, conf: float = 0.25) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """对一张 BGR 图跑推理，返回检测列表。

    返回格式: [(类别名, 置信度, (x1, y1, x2, y2)), ...]
    按置信度从高到低排序。

    提示：
        results = model.predict(source=img, conf=conf, verbose=False)
        r = results[0]
        boxes = r.boxes
        - boxes.cls  是类别 id 的 tensor
        - boxes.conf 是置信度
        - boxes.xyxy 是左上角/右下角坐标
        - r.names    是 {id: 类别名} 的字典
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    results = model.predict(source=img, conf=conf, verbose=False)
    r = results[0]
    dets: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for b in r.boxes:
        cls_id = int(b.cls[0])                       # tensor -> int
        name = r.names[cls_id]                       # id -> 类别名
        c = float(b.conf[0])                         # 置信度
        x1, y1, x2, y2 = map(int, b.xyxy[0])         # 左上/右下
        dets.append((name, c, (x1, y1, x2, y2)))
    dets.sort(key=lambda d: d[1], reverse=True)      # 按置信度降序
    return dets
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 3
def count_by_class(dets: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> Dict[str, int]:
    """按类别名计数：{"person": 4, "bus": 1}"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    counts: Dict[str, int] = {}
    for name, _, _ in dets:
        counts[name] = counts.get(name, 0) + 1
    return counts
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 4
def max_conf_target(dets, name: str):
    """返回指定类别中置信度最高的那一条；没有该类返回 None。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    best = None
    for nm, c, box in dets:
        if nm == name and (best is None or c > best[1]):
            best = (nm, c, box)
    return best
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 5
def draw_dets(img, dets, copy_img: bool = True) -> "np.ndarray":
    """把检测结果画到图上：绿框 + 顶部文字 "name conf"。

    返回画好的图；copy_img=True 时不改动原图。
    """
    import cv2
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    out = img.copy() if copy_img else img
    for name, c, (x1, y1, x2, y2) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "%s %.2f" % (name, c)
        ty = y1 - 8 if y1 - 8 > 15 else y1 + 18     # 别让文字跑出图上沿
        cv2.putText(out, label, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out
    # ↑↑↑ 你的代码写在这里 ↑↑↑
