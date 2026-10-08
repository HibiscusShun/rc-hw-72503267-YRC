# -*- coding: utf-8 -*-
"""T08 生成检测结果图：在 bus.jpg 上画 YOLO 检测框。"""
import os
import shutil
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))   # .../线上自学教案

import sys
sys.path.insert(0, os.path.join(ROOT, "07_YOLO推理"))
import hw_yolo as y                                              # noqa: E402

ASSETS = os.path.join(ROOT, "assets")

def imread_any(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

# 模型复制到英文临时路径（避免 torch 中文路径问题）
tmp = tempfile.mkdtemp(prefix="t08gen_")
pt_tmp = os.path.join(tmp, "yolov8n.pt")
shutil.copy(os.path.join(ASSETS, "yolov8n.pt"), pt_tmp)

img = imread_any(os.path.join(ASSETS, "bus.jpg"))
assert img is not None, "bus.jpg 没读到"

model = y.load_model(pt_tmp)
dets = y.detect_objects(model, img, conf=0.25)
out = y.draw_dets(img, dets)

out_path = os.path.normpath(os.path.join(HERE, "..", "截图", "T08_检测结果.png"))
okb, buf = cv2.imencode(".png", out)
if okb:
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    print("已保存:", out_path)
else:
    print("编码失败")

shutil.rmtree(tmp, ignore_errors=True)
print("检出目标:", [d[0] for d in dets])
