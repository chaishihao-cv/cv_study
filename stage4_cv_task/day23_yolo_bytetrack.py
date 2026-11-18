# day23 YOLO + ByteTrack 视频目标跟踪
# 任务：对视频逐帧检测+跟踪，输出带 ID 的可视化视频
# 思考题：跟踪 ID 跳变一般是什么原因？

from ultralytics import YOLO

# TODO: model.track(source="video.mp4", tracker="bytetrack.yaml", persist=True)
