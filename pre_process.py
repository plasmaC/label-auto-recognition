import glob
import os

from PIL import Image

size = (224, 224)
load_path = "C:/Users/fy071/Desktop/SE/cloth/img/*/*.jpg"
save_path = "C:/Users/fy071/Desktop/SE/cloth_thumbnails/"

if not os.path.exists(save_path):
    os.mkdir(save_path)

for infile in glob.glob(load_path):
    # 得到文件夹名，如Abstract_Animal_Print_Dress
    class_path = os.path.dirname(infile)
    class_name = os.path.split(class_path)[-1]

    # 得到文件名，如img_00000001.jpg
    file_name = os.path.basename(infile)

    # 如果不存在子目录则创建
    if not os.path.exists(save_path + class_name):
        os.mkdir(save_path + class_name)
        print("Processing: " + class_name)

    img = Image.open(infile)
    img.thumbnail(size, Image.ANTIALIAS)

    img.save(save_path + class_name + "/" + file_name, "JPEG")
