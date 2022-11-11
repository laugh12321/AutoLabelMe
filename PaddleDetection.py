'''
@File    :   PaddleDetection.py
@Version :   1.0
@Author  :   laugh12321
@Contact :   laugh12321@vip.qq.com
@Date    :   2022/11/11 09:24:34
@Desc    :   PaddleDetection 自动标注
'''

import os
import json

def save_labelme_results(image_list, results, labels, save_dir, threshold=0.5):
    import base64
    import io

    from PIL import Image

    def img_to_b64(image):
        f = io.BytesIO()
        image.save(f, format="PNG")
        img_bin = f.getvalue()
        return base64.encodebytes(img_bin) if hasattr(base64, "encodebytes") else base64.encodestring(img_bin)

    start_idx = 0
    for idx, imagePath in enumerate(image_list):
        image = Image.open(imagePath)
        imageWidth, imageHeight = image.size
        fileName, _ = os.path.splitext(os.path.basename(imagePath))
        annotion = {
            "version": "5.0.5", 
            "flags": {}, 
            "shapes": [], 
            "imagePath": imagePath, 
            "imageData": img_to_b64(image).decode('ascii'), 
            "imageHeight": imageHeight,
            "imageWidth": imageWidth,
        }
        
        im_bboxes_num = results['boxes_num'][idx]
        if 'boxes' in results:
            boxes = results['boxes'][start_idx:start_idx + im_bboxes_num, :].tolist()
            annotion["shapes"].extend([{
                "label": labels[int(box[0])],
                "points": [
                    [box[2], box[3]], 
                    [box[4], box[5]],
                ],
                "group_id": None, 
                "shape_type": "rectangle", 
                "flags": {}
            } for box in boxes if ((box[1] > threshold) & (box[0] > -1))])
            start_idx += im_bboxes_num

        json.dump(
            annotion, 
            open(os.path.join(save_dir, f'{fileName}.json'), "w+", encoding='utf-8'), 
            indent=4, sort_keys=False, ensure_ascii=False
        )  # 保存json