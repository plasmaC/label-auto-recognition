import glob

import vgg_imagenet
from vgg_imagenet import get_model

label_list = [
    "印花", "条纹", "印花", "无",

    "雪纺", "棉衣", "钩针编织", "牛仔", "人造", "针织", "蕾丝", "皮革", "无",

    "运动", "大衣", "衬衫", "无",

    "短袖", "无袖", "无",

    "夏装", "经典", "无"
]

model = get_model()
model.load_weights("my_model_weights_4.h5")

out_list = []


def test_img(path):
    img = vgg_imagenet.img_path_to_vec(path)
    li = model.predict(img)

    index_list = []
    for nparr in li:
        arr = nparr.tolist()[0]
        index_list.append(arr.index(max(arr)))

    index_list[1] += 4
    index_list[2] += 13
    index_list[3] += 17
    index_list[4] += 20

    out = []
    for index in index_list:
        out.append(label_list[index])

    out_list.append(out)


def predict():
    load_path = r"C:\Users\fy071\Desktop\SE\cloth\test_img\*"
    for infile in glob.glob(load_path):
        test_img(infile)
    for out in out_list:
        print(out)


def accuracy():
    true = 0
    false = 0
    for out in out_list:
        if out[0] == "印花":
            true += 1
        else:
            false += 1
    return true / (true + false)


predict()
