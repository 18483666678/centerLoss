import os
from PIL import Image
import numpy as np
from tool import utils
import traceback

label_path = "./person/list_bbox_celeba.txt"
img_path = "./person"
save_path = "../person_face"
face_size = 64
COUNT = 50

if __name__ == '__main__':
    print("gen %i image" % face_size)
    for i, line in enumerate(open(label_path)):
        if i < 2:
            continue
        try:
            strs = line.strip().split()
            image_filename = strs[0].strip()
            print(image_filename)
            person_dir = os.path.join(save_path, str(image_filename.split(".")[0]))
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

            image_file = os.path.join(img_path, image_filename)
            count = 0

            with Image.open(image_file) as img:
                img_w, img_h = img.size
                x1 = float(strs[1].strip())
                y1 = float(strs[2].strip())
                w = float(strs[3].strip())
                h = float(strs[4].strip())
                x2 = float(x1 + w)
                y2 = float(y1 + h)

                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                boxes = [[x1, y1, x2, y2]]
                _boxes = np.array(boxes)

                # 计算出人脸中心点位置
                cx = x1 + w / 2
                cy = y1 + h / 2

                # 使正样本和部分样本数量翻倍
                while count < COUNT:
                    # 让人脸中心点有少许的偏移
                    w_ = np.random.randint(-w * 0.2, w * 0.2)
                    h_ = np.random.randint(-h * 0.2, h * 0.2)
                    cx_ = cx + w_
                    cy_ = cy + h_

                    # 让人脸形成正方形，并且让坐标也有少许的偏离
                    side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                    x1_ = np.maximum(0, cx_ - side_len / 2)
                    y1_ = np.maximum(0, cy_ - side_len / 2)
                    x2_ = x1_ + side_len
                    y2_ = y1_ + side_len

                    crop_box = np.array([x1_, y1_, x2_, y2_])

                    # 计算坐标的偏移值
                    offset_x1 = (x1 - x1_) / side_len
                    offset_y1 = (y1 - y1_) / side_len
                    offset_x2 = (x2 - x2_) / side_len
                    offset_y2 = (y2 - y2_) / side_len

                    # 剪切下图片，并进行大小缩放
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                    iou = utils.iou(crop_box, _boxes, False)[0]

                    if iou > 0.5:  # positive and part
                        face_resize.save(os.path.join(person_dir, "{0}.jpg".format(count)))
                        count += 1

                    elif iou < 0.3:  # negative
                        print("Negative simples, dropout")

        except Exception as e:
            print("List empty now!")
            # traceback.print_exc()
