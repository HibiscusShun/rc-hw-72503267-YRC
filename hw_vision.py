# -*- coding: utf-8 -*-
"""
T06 传统视觉三板斧 —— 作业骨架（学生版）

三板斧：HSV 颜色阈值 → 形态学去噪 → 轮廓筛选 + 质心
四个函数对应四张图（images/ 或 assets/ 下的 quiz_0X.png）。

自检（本教案不含自动校验脚本）：自己写几行代码跑一遍 images/quiz_0X.png，
把算出来的坐标打印出来，和讲义里的真值对一下（容差 8 px）

调试利器（强烈推荐先玩这个）：
    python examples/02b_hsv_trackbar.py
    拖动滑动条找到合适的 HSV 范围，再把数值抄进这里。
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------- 工具（已给）
def _morph(mask, k=5):
    """开运算去噪 + 闭运算补洞"""
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _centroid(cnt) -> Optional[Tuple[float, float]]:
    """算轮廓质心，退化轮廓返回 None"""
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def _vertices(cnt) -> int:
    """多边形逼近后的顶点数（3=三角 4=四边 >=8≈圆）"""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    return len(approx)


def _in_range(img, lo, hi):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def _contours(mask):
    """取掩膜的所有外轮廓"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# ---------------------------------------------------------------- 1
def detect_red_circle(img) -> Optional[Tuple[float, float]]:
    """quiz_01：找出图中唯一的红色圆，返回质心 (cx, cy)；找不到返回 None。

    提示：红色的 H 在 0 附近，需要两段阈值 [0,10] 与 [170,180] 合并。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    m1 = _in_range(img, (0, 100, 50), (10, 255, 255))
    m2 = _in_range(img, (170, 100, 50), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = _morph(mask)
    contours = _contours(mask)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    return _centroid(biggest)
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 2
def detect_blue_rect(img) -> Optional[Tuple[float, float]]:
    """quiz_02：找出蓝色正方形，返回质心。蓝色 H 约 100~130。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    mask = _in_range(img, (100, 100, 50), (130, 255, 255))
    mask = _morph(mask)
    contours = _contours(mask)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    return _centroid(biggest)
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 3
def detect_green_triangle(img) -> Optional[Tuple[float, float]]:
    """quiz_03：图中有绿三角 + 红圆干扰 + 蓝块干扰。
    只返回绿色三角形的质心（用顶点数 == 3 筛选）。绿色 H 约 35~85。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    mask = _in_range(img, (35, 100, 50), (85, 255, 255))
    mask = _morph(mask)
    contours = _contours(mask)
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        if _vertices(c) == 3:
            return _centroid(c)
    return None
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 4
def detect_red_targets(img) -> List[Tuple[float, float]]:
    """quiz_04：找出图中所有红色目标的质心，按 x 升序返回。

    注意：橙色圆 (H≈19) 不是红色，别把它算进来 —— 把 H 上限开太大就会误检。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    m1 = _in_range(img, (0, 100, 50), (10, 255, 255))
    m2 = _in_range(img, (170, 100, 50), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = _morph(mask)
    contours = _contours(mask)
    centers: List[Tuple[float, float]] = []
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        cent = _centroid(c)
        if cent is not None:
            centers.append(cent)
    centers.sort(key=lambda p: p[0])   # 按 x 升序
    return centers
    # ↑↑↑ 你的代码写在这里 ↑↑↑
