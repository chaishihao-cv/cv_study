# day18 YOLOv8 训练脚本
# 任务：从 data.yaml 启动训练；观察 loss / mAP 曲线；跑预测图
# 思考题：mAP@0.5 是什么指标？

from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    # TODO: model.train(data="data.yaml", epochs=50, imgsz=640)
    pass


if __name__ == "__main__":
    main()
