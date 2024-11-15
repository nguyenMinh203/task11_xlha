import os
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
import cv2

# Đăng ký tập dữ liệu (định dạng COCO)
register_coco_instances("dog_cat_train", {}, r"C:\path\to\train_annotations.json", train_dir)
register_coco_instances("dog_cat_val", {}, r"C:\path\to\val_annotations.json", val_dir)

# Cấu hình mô hình Faster R-CNN
cfg = get_cfg()
cfg.merge_from_file("detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("dog_cat_train",)
cfg.DATASETS.TEST = ("dog_cat_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300  # Số bước huấn luyện
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Số lớp: Cat và Dog
cfg.OUTPUT_DIR = "./output"

# Huấn luyện mô hình
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Sử dụng mô hình
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Ngưỡng dự đoán
predictor = DefaultPredictor(cfg)

# Dự đoán trên ảnh
image_path = r"C:\path\to\test_image.jpg"
image = cv2.imread(image_path)
outputs = predictor(image)

# Hiển thị kết quả
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("dog_cat_val"), scale=0.8)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Result", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
