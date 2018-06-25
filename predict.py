import read_attr
from vgg_imagenet import get_model


def test_img():
    model = get_model()
    model.load_weights("my_model_weights_0.h5")
    img = read_attr.imgpath_to_vec('C:/Users/fy071/Desktop/SE/cloth/img2/Faux_Shearling_Parka-img_00000011.jpg')
    print(model.predict(img))


test_img()
