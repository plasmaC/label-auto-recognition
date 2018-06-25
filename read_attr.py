import heapq
import shutil

import numpy as np

attr_dataset_path = 'list_attr_img.txt'
attr_name_path = 'C:/Users/fy071/Desktop/SE/cloth/list_attr_cloth.txt'
attr_npy3 = 'C:/Users/fy071/Desktop/SE/cloth/data3.npy'


def traverse_attr_dataset():
    cnt = np.zeros((1000,))
    with open(attr_dataset_path, 'r') as f:
        for i, l in enumerate(f):
            if i > 1:
                ls = l.strip().split()
                path, attr = 'C:/Users/fy071/Desktop/SE/cloth/' + ls[0], ls[1:]
                n = np.array(attr, dtype='int')
                d = np.where(n == 1)
                cnt[d] = cnt[d] + 1
                if i % 10000 == 0:
                    print(d)


def save_dataset_to_npy():
    data = np.zeros((289222, 1000))
    with open(attr_dataset_path, 'r') as f:
        for i, l in enumerate(f):
            if i > 1:
                ls = l.strip().split()
                attr = ls[1:]
                data[i - 2] = np.array(attr, dtype='int')
                if i % 10000 == 0:
                    print(i)
    np.save('attr_data.npy', data)


def count_max_n_attr(n=20):
    attr_name = []

    max_n_attr = []
    with open(attr_name_path, 'r') as f:
        for i, l in enumerate(f):
            if i > 1:
                ls = l.strip().split()
                attr_name.append((ls[0], ls[1]))

    cnt = np.load('cnt.npy')
    c = heapq.nlargest(n, range(len(cnt)), cnt.take)
    for ci in c:
        max_n_attr.append((ci, cnt[ci], *attr_name[ci]))
    max_n_attr = sorted(max_n_attr, key=lambda l: l[0])
    max_n_idx = np.array([item[0] for item in max_n_attr])
    print(max_n_idx)
    return max_n_idx, max_n_attr


def extract_img():
    # 抽取的标签(col)
    max_20_idx, _ = count_max_n_attr()

    data = np.load('attr_data')
    n = 289222
    cnt = 0

    # 抽取的数据idx(row)

    with open(attr_dataset_path, 'r') as reader, open('C:/Users/fy071/Desktop/SE/cloth/list_attr_img2.txt',
                                                      'w') as writer:
        for i, l in enumerate(reader):
            if i > 1:
                ls = l.strip().split()
                path = 'C:/Users/fy071/Desktop/SE/cloth/' + ls[0]

                lp = np.where(data[i - 2][max_20_idx] == 1)
                if lp[0].shape[0] != 0:
                    path_rear = '-'.join(ls[0].split('/')[1:])
                    new_path = 'C:/Users/fy071/Desktop/SE/cloth/img2/' + path_rear
                    shutil.copyfile(path, new_path)

                    writer.write(new_path + '\n')

                    # 把图片复制到img2


def extract_npy():
    max_20_idx, _ = count_max_n_attr()
    data = np.load('attr_data')
    n = 289222
    cnt = 0
    # 抽取的数据idx(row)
    ls = []
    for i in range(n):
        lp = np.where(data[i][max_20_idx] == 1)
        if lp[0].shape[0] != 0:
            ls.append(i)

    ls = np.array(ls)
    data2 = data[ls, :][:, max_20_idx]

    np.save('idx2.npy', ls)
    print(data2.shape)


def convert_to_zero():
    data2 = np.load('data2.npy')
    data3 = np.where(data2 == -1, 0, 1)
    print(data3.shape)
    np.save('data3.npy', data3)


def tune_data():
    data3 = np.load(attr_npy3)
    striped = 18
    stripe = 17
    floral_print = 8
    _print = 13
    floral = 7
    print(data3.shape)
    for i in range(len(data3)):
        data3[i, _print] = max(data3[i, _print], data3[i, floral_print])
        data3[i, floral] = max(data3[i, floral], data3[i, floral_print])
        data3[i, stripe] = max(data3[i, stripe], data3[i, striped])
    data4 = data3[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]]
    np.save('data4.npy', data4)


def convert_validation():
    data4 = np.load('data4.npy')
    sp = 20

    data_test = []
    data_train = []

    with open('list_attr_img2.txt', 'r') as reader, open('list_attr_test.txt', 'w') as writer1, open(
            'list_attr_train.txt', 'w') as writer2:
        for i, line in enumerate(reader):
            if i % 20 == 0:
                writer1.write(line)
                data_test.append(data4[i])
            else:
                writer2.write(line)
                data_train.append(data4[i])
    np.save('data_test.npy', np.array(data_test))
    np.save('data_train.npy', np.array(data_train))


def split_train():
    with open('list_attr_train.txt', 'r') as reader:
        lines = []
        for i, line in enumerate(reader):
            lines.append(line)
            if i % 2000 == 1999:
                path = 'list_attr_train' + str(int(i // 2000)) + '.txt'
                with open(path, 'w') as writer:
                    writer.writelines(lines)
                lines.clear()


# tensor = to_tensor("train/list_attr_train0.txt")
# print(tensor.shape)


def analyse():
    data = np.load('data_test.npy')
    perm = [7, 16, 12, 0, 2, 3, 4, 5, 8, 9, 10, 6, 11, 13, 14, 15, 17, 1]
    data[:, range(18)] = data[:, perm]
    np.save('data_test2.npy', data)


def analyse2():
    cnt = 0
    data = np.load('data_test2.npy')
    datas = (data[:, :3], data[:, 3:11], data[:, 11:14], data[:, 14:16], data[:, 16:18])
    for i in range(len(data)):
        for j in range(5):
            v = np.count_nonzero(datas[j][i])
            if v > 1:
                print(i, j, v)
                cnt += 1
    print(cnt)


def eras():
    data = np.load('data_test2.npy')
    datas = (data[:, :3], data[:, 3:11], data[:, 11:14], data[:, 14:16], data[:, 16:18])

    data2 = []

    with open('list_attr_test.txt', 'r') as reader, open(
            'list_attr_test2.txt', 'w') as writer:
        for i, line in enumerate(reader):
            for j in range(5):
                if np.count_nonzero(datas[j][i]) < 1:
                    writer.write(line)
                    data2.append(data[i])
                    break

    np.save('data_test3.npy', np.array(data2))


def addd():
    data = np.load('data_test.npy')
    datas = (data[:, :3], data[:, 3:11], data[:, 11:14], data[:, 14:16], data[:, 16:18])
    # 18-23
    new_data = np.zeros(shape=(data.shape[0], data.shape[1] + 5))
    new_data[:, 0:3], new_data[:, 4:12], new_data[:, 13:16], new_data[:, 17:19], new_data[:, 20:22] = data[:, :3], data[
                                                                                                                   :,
                                                                                                                   3:11], data[
                                                                                                                          :,
                                                                                                                          11:14], data[
                                                                                                                                  :,
                                                                                                                                  14:16], data[
                                                                                                                                          :,
                                                                                                                                          16:18]

    label = [3, 12, 16, 19, 22]

    for i in range(len(data)):

        for j in range(5):
            if np.count_nonzero(datas[j][i]) == 0:
                new_data[i, label[j]] = 1
    print(new_data.shape)
    np.save('data_test2.npy', np.array(new_data))


def analyse3():
    cnt = 0
    data = np.load('data_train.npy')
    for i in range(len(data)):
        v = np.count_nonzero(data[i])
        if v > 1:
            cnt += 1
    print(cnt)


def all_shuff():
    data = np.load('data_train.npy')
    perm = np.random.permutation(range(len(data)))
    data = data[perm, :]
    np.save('data_train2.npy', data)
    with open('list_attr_train.txt', 'r') as reader, open('list_attr_train2.txt', 'w') as writer:
        list_train = []

        for line in reader:
            list_train.append(line)

        for i in perm:
            writer.write(list_train[i])


def re_extract_img():
    with open('list_attr_train.txt', 'r') as reader, open('list_attr_train2.txt', 'w') as writer:
        for i, line in enumerate(reader):
            path_rear = line.strip().split('/')[-1]
            new_path = 'C:/Users/fy071/Desktop/SE/cloth/train_img/' + path_rear

            shutil.copyfile(line[:-1], new_path)
            writer.write(new_path + '\n')

    with open('list_attr_test.txt', 'r') as reader, open('list_attr_test2.txt', 'w') as writer:
        for i, line in enumerate(reader):
            path_rear = line.strip().split('/')[-1]
            new_path = 'C:/Users/fy071/Desktop/SE/cloth/test_img/' + path_rear
            shutil.copyfile(line[:-1], new_path)
            writer.write(new_path + '\n')


def rename_path_in_train_txt():
    for i in range(80):
        with open('train/list_attr_train' + str(i) + '.txt', 'r') as reader, open(
                'train/list_attr_train2_' + str(i) + '.txt', 'w') as writer:
            for i in reader:
                path_rear = i.split('/')[-1]
                new_path = 'C:/Users/fy071/Desktop/SE/cloth/train_img/' + path_rear
                writer.write(new_path)


rename_path_in_train_txt()
