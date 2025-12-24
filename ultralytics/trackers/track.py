# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial
from pathlib import Path

import torch

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker     #带上点就是相对导入

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:    #-> None  表示不返回值
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool): Whether to persist the trackers if they already exist.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.

    Examples:
        Initialize trackers for a predictor object:
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if hasattr(predictor, "trackers") and persist:       #hasattr 是 Python 的内置函数，用于判断一个对象是否包含指定名称的属性，
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:    #路径是这个/home/waas/yolo11_track/ultralytics/cfg/trackers/botsort.yaml 里面有tracker_type属性
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    trackers = []
    for _ in range(predictor.dataset.bs):        #predictor.dataset.bs 是批量大小（一次处理的图像 / 帧数量）
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes.  #判断是否实时流
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video #
    #初始化视频路径列表：vid_path 用于记录每个批次对应的视频路径，后续用于判断是否切换到新视频（如果切换，需要重置跟踪器）。

def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    path, im0s = predictor.batch[:2]          ##path：当前批次图像 / 帧的文件路径；im0s：原始图像（未预处理的图像，用于跟踪器获取图像信息）。
    ##切片操作，获取前两项

    is_obb = predictor.args.task == "obb"          ##是否为旋转框
    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]   #非流模式下，所有帧共用第 1 个跟踪器（0 索引）
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:   #重置跟踪器（当切换视频时）：
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()  #如果是 obb 任务，取旋转框结果（obb）；否则取普通矩形框结果（boxes）。
        if len(det) == 0:   #跳过无检测结果的帧：如果当前帧没有检测到物体，无需更新跟踪器，直接进入下一轮循环。
            continue
        tracks = tracker.update(det, im0s[i])  #更新轨迹
        if len(tracks) == 0:       #跳过无跟踪结果的帧：如果跟踪器没有输出（如所有检测物体都无法匹配历史轨迹），直接进入下一轮。
            continue
        idx = tracks[:, -1].astype(int)   #提取跟踪结果中最后一列的元素，并将其转换为整数
        predictor.results[i] = predictor.results[i][idx]  #通过 idx 筛选出有跟踪结果的检测框（去除未被跟踪的检测框），更新当前帧的预测结果

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:   
    """
    Register tracking callbacks to the model for object tracking during prediction.
    函数作用：将上述两个回调函数（on_predict_start 和 on_predict_postprocess_end）注册到 YOLO 模型中，使模型在预测过程中自动执行跟踪逻辑。
    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
    
    """
    这段代码的核心逻辑是：
    通过 TRACKER_MAP 关联跟踪器类型与实现类；
    定义 on_predict_start 在预测开始时初始化跟踪器（根据配置选择 ByteTrack 或 BOTSORT）；
    定义 on_predict_postprocess_end 在检测后用跟踪器更新结果（绑定跟踪 ID、处理视频切换）；
    通过 register_tracker 将上述逻辑注册到模型，使 YOLO 检测与目标跟踪功能无缝结合。
    最终实现：在视频 / 流预测时，模型能自动为每个物体分配唯一跟踪 ID，并在连续帧中追踪其运动。
    """