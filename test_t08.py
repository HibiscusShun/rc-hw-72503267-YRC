# -*- coding: utf-8 -*-
"""T08 自检：加载 yolov8n.pt，在 bus.jpg 上验证 5 个函数。"""
import os
import shutil
import tempfile

import cv2
import numpy as np

import hw_yolo as y

ASSETS = r"C:\Users\86185\Documents\xwechat_files\wxid_kxs4afbvpis132_42ba\temp\RWTemp\2026-09\be1e11229f69294bd588ea0fb12c30be\线上自学教案_解压\线上自学教案\assets"

def imread_any(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

# 1. 把 yolov8n.pt 复制到英文临时路径，加载
tmp = tempfile.mkdtemp(prefix="t08_")
pt_tmp = os.path.join(tmp, "yolov8n.pt")
shutil.copy(os.path.join(ASSETS, "yolov8n.pt"), pt_tmp)
model = y.load_model(pt_tmp)
print("模型已加载:", type(model).__name__)

# 2. 读 bus.jpg 并推理
img = imread_any(os.path.join(ASSETS, "bus.jpg"))
assert img is not None, "bus.jpg 没读到"
dets = y.detect_objects(model, img, conf=0.25)

ok = True
def check(name, cond, detail=""):
    global ok
    if not cond:
        ok = False
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

names = [d[0] for d in dets]
print("检出", len(dets), "个目标:", names)

# 通过标准：检出 person 和 bus
check("检出 person", "person" in names, "有 person")
check("检出 bus", "bus" in names, "有 bus")

# 格式：每个是 (name, conf, (x1,y1,x2,y2))
fmt_ok = all(isinstance(d[0], str) and 0.0 <= d[1] <= 1.0
             and len(d[2]) == 4 and all(isinstance(v, int) for v in d[2])
             for d in dets)
check("格式正确 / 置信度合法", fmt_ok, "")

# 有序：置信度降序
sorted_ok = all(dets[i][1] >= dets[i + 1][1] for i in range(len(dets) - 1))
check("按置信度降序", sorted_ok, "")

# 3. 类别计数
counts = y.count_by_class(dets)
check("count_by_class", "person" in counts and "bus" in counts, str(counts))

# 4. 取 person 最高置信度
best = y.max_conf_target(dets, "person")
check("max_conf_target person", best is not None and best[0] == "person" and best[1] == max(d[1] for d in dets if d[0] == "person"), f"{best[0]} conf={best[1]:.2f}")
check("max_conf_target 无该类=None", y.max_conf_target(dets, "cat") is None, "")

# 5. draw_dets
out = y.draw_dets(img, dets)
check("draw_dets 画框且不改原图", out.shape == img.shape and not np.array_equal(out, img), f"{out.shape}")

shutil.rmtree(tmp, ignore_errors=True)
print()
print("==== 全部通过 ✅ ====" if ok else "==== 有失败项 ❌ ====")
