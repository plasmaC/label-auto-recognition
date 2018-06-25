import tensorflow as tf
from keras import metrics
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import *
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras_preprocessing import image

K.set_image_data_format('channels_last')
data_train = np.load('data_train.npy')
data_w = np.count_nonzero(data_train, axis=0) / len(data_train)


def img_path_to_vec(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def to_tensor(path):
    tensor = np.zeros((2000, 224, 224, 3), dtype=int)
    with open(path, 'r') as reader:
        for i, line in enumerate(reader):
            img_vector = img_path_to_vec(line[:-1])
            tensor[i] = img_vector
    return tensor


def fashion_loss(y_true, y_pred):
    return -data_w * y_true * K.log(y_pred) - (1 - data_w) * (1 - y_true) * K.log(1 - y_pred)


def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)


def get_model():
    attr_count = {
        'attr1': 3 + 1,
        'attr2': 8 + 1,
        'attr3': 3 + 1,
        'attr4': 2 + 1,
        'attr5': 2 + 1,
    }

    x_input = Input(shape=(224, 224, 3))

    base_model = InceptionV3(input_tensor=x_input, weights='imagenet', include_top=False)
    for i in range(len(base_model.layers) - 3):
        base_model.layers[i].trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in attr_count.items()]

    model = Model(inputs=base_model.input, outputs=x)

    adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(adam, loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

    return model


def train(start_training=0):
    model = get_model()

    # 读取模型权重
    if start_training != 0:
        model.load_weights("my_model_weights_" + str(start_training - 1) + ".h5")

    tensorboard = TensorBoard(log_dir="log", histogram_freq=1, write_graph=True)

    ys = np.load("data_train.npy")
    for j in range(start_training, 6):
        for i in range(80):
            print("===" + str(i) + "=========")
            x = to_tensor("train/list_attr_train2_" + str(i) + ".txt")
            y = ys[2000 * i:2000 * (i + 1)]
            y = [y[:, :4], y[:, 4:13], y[:, 13:17], y[:, 17:20], y[:, 20:23]]
            model.fit(x=x, y=y, batch_size=100, epochs=1, shuffle=True, validation_split=0.05, callbacks=[tensorboard])

        model.save_weights('my_model_weights_' + str(j) + '.h5')


train(start_training=0)


def test_img():
    model = get_model()
    model.load_weights("my_model_weights_78.h5")

    ys = np.load("data_test.npy")
    cnt = [0] * 5
    with open('list_attr_test2.txt', 'r') as reader:
        for i, line in enumerate(reader):

            idx = np.where(ys[i] == 1)[0]
            img = img_path_to_vec(line[:-1])
            p = (model.predict(img))
            l = []
            for pi in p:
                for j in range(len(pi[0])):
                    l.append(1 if pi[0][j] > 0.1 else 0)
            l = np.array(l)

            for j in range(5):
                if l[idx[j]] == 1:
                    cnt[j] += 1

            if i % 100 == 0:
                print(i, cnt)

    for j in range(5):
        cnt[j] / len(ys)
    print(cnt)
