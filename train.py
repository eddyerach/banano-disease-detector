import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog


folder_name='consolidado_v2_a/consolidado'


import os
import numpy as np
import json
from detectron2.structures import BoxMode
from os.path import exists



def get_dataset_dicts(directory):
    classes = ['disease']
    dataset_dicts = []
    for idx, filename in enumerate([file for file in os.listdir(directory) if file.endswith('.json')]):
        #print(f'idx: {idx}, filename: {filename}')
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        #print(f'Value: {img_anns["imagePath"]}')
        filename = os.path.join(directory, img_anns["imagePath"])
        #print('FILENAME:', '/banano/uvas/racimos/datasets_entrenamiento/'+filename)
        img_path = '/banano/uvas/racimos/datasets_entrenamiento/'+filename

        if not exists(img_path):
            extension = filename.split('.')[-1]
            if extension == 'JPG' or extension == 'jpg':  
                filename = ('.').join(filename.split('.')[:-1]) + '.png'

        img_path = '/banano/uvas/racimos/datasets_entrenamiento/'+filename


        img = cv2.imread(img_path)
        print(f'img_path: {img_path}')
        record["file_name"] = filename
        #print(f'idx: {idx}, filename: {filename}')
        record["image_id"] = idx
        record["height"] = img.shape[0]
        record["width"] = img.shape[1]
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    #print(f'{dataset_dicts}')
    return dataset_dicts

dicts = get_dataset_dicts(folder_name+'/')

#print(dicts)



for d in ["train", "test"]:
    
    DatasetCatalog.register("dataset_" + d, lambda d=d: get_dataset_dicts(folder_name+'/' + d))
    MetadataCatalog.get("dataset_" + d).set(thing_classes=['disease'])

dataset_metadata = MetadataCatalog.get("dataset_train")



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 4000
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

print(f'checkpoint dir: {cfg.OUTPUT_DIR}')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)

trainer.train()
