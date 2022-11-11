'''
@File    :   yolov5.py
@Version :   1.0
@Author  :   laugh12321
@Contact :   laugh12321@vip.qq.com
@Date    :   2022/11/11 09:26:48
@Desc    :   yolov5 自动标注
'''

from pathlib import Path


def save_labelme_results(image_path, det, names, save_dir):
    import base64
    import io
    import json

    from PIL import Image

    def img_to_b64(image):
        f = io.BytesIO()
        image.save(f, format="PNG")
        img_bin = f.getvalue()
        return base64.encodebytes(img_bin) if hasattr(base64, "encodebytes") else base64.encodestring(img_bin)

    image = Image.open(image_path)
    imageWidth, imageHeight = image.size
    imagePath, fileName = str(image_path), image_path.stem

    annotion = {
        "version": "5.0.5", 
        "flags": {}, 
        "shapes": [], 
        "imagePath": imagePath, 
        "imageData": img_to_b64(image).decode('ascii'), 
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
    }

    annotion["shapes"].extend([{
        "label": names[int(cls_id)], # clss label
        "points": [[int(xyxy[0]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[3])]], # bounding box
        "group_id": None, 
        "shape_type": "rectangle", 
        "flags": {}
    } for *xyxy, _, cls_id in reversed(det)])

    json.dump(
        annotion, 
        open(Path(save_dir, f'{fileName}.json'), "w+", encoding='utf-8'), 
        indent=4, sort_keys=False, ensure_ascii=False
    )  # 保存json