import json, os
from collections import defaultdict
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

def read_image_info_and_annotations(annotation_path):
    annotations_json, annotations = json.load(open(annotation_path)), defaultdict(dict)
    for img in annotations_json["images"]: 
        annotations[img["id"]]["img_info"] = img
        annotations[img["id"]]["instances"] = list()
    for anno in annotations_json["annotations"]: 
        annotations[anno["image_id"]]["instances"].append(anno)
    return annotations

def load_pedestrian_instances(dirname: str, split: str):
    dicts = list()
    annotations = read_image_info_and_annotations(os.path.join(dirname, "annotations", f"{split}.json"))
    for img_dict in annotations.values():
        r = {
            "file_name": os.path.join(dirname, "images", img_dict["img_info"]["file_name"]),
            "image_id": img_dict["img_info"]["id"],
            "height": img_dict["img_info"]["height"],
            "width": img_dict["img_info"]["width"],
            "annotations": list()
        } 

        for instance in img_dict["instances"]:
            if ("ignore" in instance and not instance["ignore"]) or (not instance["iscrowd"]):
                r["annotations"].append({
                    "category_id": 1,
                    "bbox": instance["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS
                })
        
        if not r["annotations"]:
            r["annotations"].append({
                "category_id": 0,
                "bbox": [0., 1., 2., 3.],
                "bbox_mode": BoxMode.XYWH_ABS
            })

        dicts.append(r)
    return dicts

def register_pedestrian_dataset(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_pedestrian_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=["_background", "pedestrian"], dirname=dirname, split=split)

def register_all_pedestrian_datasets(root):
    SPLITS = [
        ("caltech_pedestrians_train", "Caltech_Pedestrians", "train"),
        ("caltech_pedestrians_val", "Caltech_Pedestrians", "val"),
        ("caltech_pedestrians_test", "Caltech_Pedestrians", "test"),
        ("eurocity_train", "EuroCity", "train"),
        ("eurocity_val", "EuroCity", "val")
    ]
    for name, dirname, split in SPLITS:
        register_pedestrian_dataset(name, os.path.join(root, dirname), split),
        MetadataCatalog.get(name).evaluator_type = "pedestrian"

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pedestrian_datasets(_root)