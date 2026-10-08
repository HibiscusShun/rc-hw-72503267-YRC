# -*- coding: utf-8 -*-
"""
T07 进阶 · 完整实时追踪程序（全部自己写）
========================================
这一题只给需求与思路，**不给代码框架**——它是把 T06/T07 学过的东西
串成"一个真正能用的程序"，也是电赛现场最需要的那种程序。

需求
----
  1. 打开视频源，实时识别**指定颜色**图案的质心，并在画面上实时显示：
       · 质心画十字 + 圆圈，旁边写实时坐标 (x, y)
       · 左上角显示"当前在追什么颜色"与 FPS
  2. 按 1 / 2 / 3 / 4 切换追踪颜色：1=红 2=蓝 3=绿 4=黄；按 q 退出
  3. 打印 patterns.png（红圆 / 蓝方块 / 绿三角 / 黄圆）举到镜头前即可验证

视频源（三种任一，按你的条件选）
------------------------------
  · 网络视频流：cv2.VideoCapture("http://IP:4747/video")   ← **默认就用这条**
  · USB 摄像头 / 实体摄像头：cv2.VideoCapture(0)（传参数 usb）
  · 什么都没有：先用 cv2.imread("images/patterns.png") 当一帧，把识别逻辑调对，
      最后再接真实视频源。**线上自学完全允许只用图片/视频文件交差。**
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
from vision_util import setup, imread, FPSMeter, draw_point, put_hud   # noqa: E402

setup()

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 可改的配置
# 颜色名 -> 它属于第几个数字键（1/2/3/4）
COLOR_KEYS = {"red": "1", "blue": "2", "green": "3", "yellow": "4"}

# 手机 DroidCam 的 IP（换网络就要改），端口固定 4747
DROIDCAM_IP = "192.168.1.100"
DROIDCAM_URL = "http://%s:4747/video" % DROIDCAM_IP

# 没有摄像头时用哪张图/视频练手（指向教案 images/patterns.png）
FALLBACK_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "06_传统视觉", "images", "patterns.png")

# 各颜色的 HSV 阈值（OpenCV H 范围 0~179；S≥100、V≥60）
# 红色 H 是环形的，需要 [0,10] 和 [170,180] 两段，每段是一个 (lo, hi)
COLOR_HSV = {
    "red":    [((0, 100, 60), (10, 255, 255)), ((170, 100, 60), (180, 255, 255))],
    "blue":   [((100, 100, 60), (130, 255, 255))],
    "green":  [((35, 100, 60), (85, 255, 255))],
    "yellow": [((20, 100, 60), (35, 255, 255))],
}


# ---------------------------------------------------------------- 必须实现
def detect_color(frame, color_name):
    """在 BGR 帧里找出 color_name 对应颜色目标的质心。

    :param frame: BGR 图（单帧）
    :param color_name: "red" / "blue" / "green" / "yellow"
    :return: (cx, cy) 或 None（没找到）

    提示：转 HSV → inRange（红色两段合并）→ 形态学开+闭 → 最大轮廓 → moments
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in COLOR_HSV[color_name]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    # 形态学：开运算去白点 + 闭运算补黑洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 找最大轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])
    # ↑↑↑ 你的代码写在这里 ↑↑↑


def open_source(src):
    """打开视频源并做低延迟设置，返回 cv2.VideoCapture（打不开也要返回对象）。

    src 可能是 0（摄像头）、"http://..."（网络流）或一个视频文件路径。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 网络流默认缓冲几十帧，压到 1 帧降延迟
    return cap
    # ↑↑↑ 你的代码写在这里 ↑↑↑


def resolve_source(arg):
    """把命令行参数解析成 cv2.VideoCapture 能吃的源。

    - 无参数      -> 默认 DroidCam 网络流
    - "usb"       -> 本机摄像头 0
    - "1.2.3.4"   -> 手机 IP，拼成 DroidCam URL
    - 其它字符串   -> 当作视频/图片文件路径
    """
    if not arg:
        return DROIDCAM_URL
    if arg.lower() == "usb":
        return 0
    if arg.replace(".", "").isdigit():          # 是 IP
        return "http://%s:4747/video" % arg
    return arg                                  # 文件路径


def main():
    """主循环：读帧 → 检测当前颜色 → 画 HUD → 按键切换/退出。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    KEY = {"1": "red", "2": "blue", "3": "green", "4": "yellow"}

    src = resolve_source(sys.argv[1] if len(sys.argv) > 1 else None)
    cap = open_source(src)
    if not cap.isOpened():
        print("[警告] 打不开视频源 %r，改用手册给的图片当一帧演示" % (src,))
        frame = imread(FALLBACK_IMAGE)
        if frame is None:
            print("连 patterns.png 也没有，退出。")
            return

    current = "red"
    meter = FPSMeter()

    while True:
        if cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                # 断流重连：释放 -> 等 0.2s -> 重新打开，别直接崩
                print("断流，0.2s 后重连...")
                cap.release()
                time.sleep(0.2)
                cap = open_source(src)
                continue
        else:
            frame = imread(FALLBACK_IMAGE)   # 图片源：每帧重新读当"视频"

        pt = detect_color(frame, current)
        if pt:
            cx, cy = pt
            frame = draw_point(frame, cx, cy, text="(%.0f, %.0f)" % (cx, cy))

        fps = meter.tick()
        frame = put_hud(frame, ["Tracking: " + current, "FPS: %.1f" % fps])

        cv2.imshow("track", frame)
        key = cv2.waitKey(1) & 0xFF
        if key != 255 and chr(key) in KEY:
            current = KEY[chr(key)]
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    # ↑↑↑ 你的代码写在这里 ↑↑↑


if __name__ == "__main__":
    main()
