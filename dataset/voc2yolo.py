import xml.etree.ElementTree as ET
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)

dataset = Path(__file__).parent
images_dir = dataset / "images"
anns_dir = dataset / "annotations"
labels_dir = dataset / "labels"
labels_dir.mkdir(exist_ok=True)

CLASS_MAP = {"pothole": 0}

xml_files = sorted(anns_dir.glob("*.xml"))
random.seed(42)
random.shuffle(xml_files)

split = int(len(xml_files) * 0.8)
train_files = xml_files[:split]
val_files = xml_files[split:]

for split_name, xml_list in [("train", train_files), ("val", val_files)]:
    out_dir = dataset / split_name / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir = dataset / split_name / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    for xml_path in xml_list:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        filename = root.find("filename").text

        img_path = images_dir / filename
        if img_path.exists():
            import shutil
            shutil.copy2(img_path, img_out_dir / filename)

        lines = []
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls]
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        txt_path = out_dir / xml_path.with_suffix(".txt").name
        txt_path.write_text("\n".join(lines))

log.info("Train: %d, Val: %d", len(train_files), len(val_files))