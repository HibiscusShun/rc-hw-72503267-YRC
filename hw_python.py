# -*- coding: utf-8 -*-
"""
T03 Python 速通 —— 作业骨架（学生版，不含答案）

要求：把 6 个函数的 TODO 部分补全，不要改函数名/参数/返回值类型。
自检（本教案不含自动校验脚本）：自己写几行 print 调用这六个函数，看输出对不对

卡住时把报错整段丢给 AI（见 00-3），但请自己先看懂它改了什么。
"""
from typing import Dict, List, Tuple


# ---------------------------------------------------------------- 1
def count_colors(colors: List[str]) -> Dict[str, int]:
    """统计列表中每种颜色出现的次数。

    输入: ["red", "blue", "red", "green", "red"]
    输出: {"red": 3, "blue": 1, "green": 1}
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    result: Dict[str, int] = {}
    for c in colors:
        result[c] = result.get(c, 0) + 1
    return result
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 2
def largest_box(boxes: List[Tuple[float, float]]) -> int:
    """给定 [(宽, 高), ...]，返回面积最大的那个的下标。空列表返回 -1。

    输入: [(10, 20), (30, 5), (12, 12)]
    输出: 0            # 10*20=200 最大
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    if not boxes:
        return -1
    best_i = 0
    best_area = boxes[0][0] * boxes[0][1]
    for i, (w, h) in enumerate(boxes):
        area = w * h
        if area > best_area:
            best_area = area
            best_i = i
    return best_i
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 3
def filter_by_conf(dets: List[Tuple[str, float]], thr: float = 0.5) -> List[Tuple[str, float]]:
    """过滤检测结果，只保留置信度 >= thr 的，保持原顺序。

    输入: [("person", 0.92), ("bus", 0.31), ("car", 0.55)], thr=0.5
    输出: [("person", 0.92), ("car", 0.55)]
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    return [(name, conf) for name, conf in dets if conf >= thr]
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 4
def clamp(value: float, lo: float, hi: float) -> float:
    """把 value 限制在 [lo, hi] 区间内（控制量限幅，闭环里天天用）。

    clamp(1.5, -1, 1) -> 1.0
    clamp(-3,  -1, 1) -> -1.0
    clamp(0.2, -1, 1) -> 0.2
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 5
def moving_average(values: List[float], k: int = 3) -> List[float]:
    """滑动平均（窗口 k），输出长度与输入相同；前 k-1 项用"到目前为止的均值"。

    输入: [1, 2, 3, 4, 5], k=3
    输出: [1.0, 1.5, 2.0, 3.0, 4.0]
          # 第1个: 1   第2个: (1+2)/2
          # 第3个: (1+2+3)/3   第4个: (2+3+4)/3   第5个: (3+4+5)/3
    k <= 1 时原样返回。
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    if k <= 1:
        return list(values)
    result: List[float] = []
    run_sum = 0.0
    for i, v in enumerate(values):
        run_sum += v
        if i >= k:
            run_sum -= values[i - k]
        win_len = min(i + 1, k)
        result.append(run_sum / win_len)
    return result
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 6
def parse_detection_line(line: str) -> Tuple[str, float, Tuple[int, int, int, int]]:
    """解析一行检测结果文本，返回 (类别名, 置信度, (x1,y1,x2,y2))。

    输入: "person 0.92 100 120 300 400"
    输出: ("person", 0.92, (100, 120, 300, 400))
    格式不对时返回 ("", 0.0, (0, 0, 0, 0))
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    parts = line.split()
    if len(parts) != 6:
        return ("", 0.0, (0, 0, 0, 0))
    name = parts[0]
    try:
        conf = float(parts[1])
        x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
    except ValueError:
        return ("", 0.0, (0, 0, 0, 0))
    return (name, conf, (x1, y1, x2, y2))
    # ↑↑↑ 你的代码写在这里 ↑↑↑
