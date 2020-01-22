# zip形式のデータから、使えそうなデータのみを抽出する。
# おそらくデモしか使用しない。pathとか再利用とか考えていない。

import os
import glob

ORIGIN_DATA_DIR = './static/origindata'
WORK_DIR = './static/data/img'

data_dirs = os.listdir(ORIGIN_DATA_DIR)



def mkdir_each_target(item_id):
    dir_path = os.path.join(WORK_DIR, item_id)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        return dir_path
    return False

def move_image(data, item_id):
    name = data.split('/')[-1]
    ignore_names = [f'_{i}' for i in range(2, 10)] # 拡大画像や背面画像、タグ画像を除外
    for i in ignore_names:
        if i in name:
            return
    os.rename(data, os.path.join(WORK_DIR, item_id, name))

def main():
    for d in data_dirs:
        fp = os.path.join(ORIGIN_DATA_DIR, d, '*.JPG')
        images = glob.glob(fp)

        items = []
        for image in images:
            name = image.split('/')[-1]
            if '_' in name:
                continue
            item = name.rstrip('.JPG')
            mkdir_each_target(item)
            items.append(item)

        for item in items:
            for img in images:
                if item in img:
                    move_image(img, item)

if __name__ == "__main__":
    main()


