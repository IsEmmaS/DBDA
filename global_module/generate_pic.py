import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
from Utils import extract_samll_cubic
import torch.utils.data as Data
import time


def load_dataset(Dataset):
    """
    Read the dataset by the index of the string:Dataset, abs path.
    Args:
        Dataset(string): 'IN', 'UP', 'PC', 'SV', 'KSC', 'BS', 'DN', 'DN_1', 'WHL', 'HC', 'HH'
    Returns:
        truple: data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT
    
    """
    if Dataset == 'IN':
        mat_data = sio.loadmat('/root/DBDA/datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('/root/DBDA/datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        uPavia = sio.loadmat('/root/DBDA/datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('/root/DBDA/datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        # VALIDATION_SPLIT = 0.98
        VALIDATION_SPLIT = 0.91
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PC':
        uPavia = sio.loadmat('/root/DBDA/datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('/root/DBDA/datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.999
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'SV':
        SV = sio.loadmat('/root/DBDA/datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('/root/DBDA/datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.98
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        KSC = sio.loadmat('/root/DBDA/datasets/KSC.mat')
        gt_KSC = sio.loadmat('/root/DBDA/datasets/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        # VALIDATION_SPLIT = 0.95
        VALIDATION_SPLIT = 0.91
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'BS':
        BS = sio.loadmat('/root/DBDA/datasets/Botswana.mat')
        gt_BS = sio.loadmat('/root/DBDA/datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'DN':
        DN = sio.loadmat('/root/DBDA/datasets/Dioni.mat')
        gt_DN = sio.loadmat('/root/DBDA/datasets/Dioni_gt_out68.mat')
        data_hsi = DN['ori_data']
        gt_hsi = gt_DN['map']
        TOTAL_SIZE = 20024
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'DN_1':
        DN_1 = sio.loadmat('/root/DBDA/datasets/DN_1/Dioni.mat')
        gt_DN_1 = sio.loadmat('/root/DBDA/datasets/DN_1/Dioni_gt_out68.mat')
        data_hsi = DN_1['imggt']
        gt_hsi = gt_DN_1['map']
        TOTAL_SIZE = 20024
        VALIDATION_SPLIT = 0.98
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'WHL':
        WHL = sio.loadmat('/root/DBDA/datasets/WHL/WHU_Hi_LongKou.mat')
        gt_WHL = sio.loadmat('/root/DBDA/datasets/WHL/WHU_Hi_LongKou_gt.mat')
        data_hsi = WHL['WHU_Hi_LongKou']
        gt_hsi = gt_WHL['WHU_Hi_LongKou_gt']
        TOTAL_SIZE = 204542
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'HC':
        HC = sio.loadmat('/root/DBDA/datasets/HC/WHU_Hi_HanChuan.mat')
        gt_HC = sio.loadmat('/root/DBDA/datasets/HC/WHU_Hi_HanChuan_gt.mat')
        data_hsi = HC['WHU_Hi_HanChuan']
        gt_hsi = gt_HC['WHU_Hi_HanChuan_gt']
        TOTAL_SIZE = 257530
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'HH':
        HH = sio.loadmat('/root/DBDA/datasets/HH/WHU_Hi_HongHu.mat')
        gt_HH = sio.loadmat('/root/DBDA/datasets/HH/WHU_Hi_HongHu_gt.mat')
        data_hsi = HH['WHU_Hi_HongHu']
        gt_hsi = gt_HH['WHU_Hi_HongHu_gt']
        TOTAL_SIZE = 386693
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def save_cmap(img, cmap, fname):
    """
    Saves an image with a specified colormape to a file.

    Args:
        img (numpy.ndarray): The input image to be saved with a colormap.
        cmap (str or matplotlib.colors.Colormap): The colormap to be applied to the image. Can be a string (e.g., 'viridis', 'plasma')
            or a matplotlib.colors.Colormap object.
        fname (str): The filename to save the image to.

    Returns:
        None
     Example:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100)
        >>> save_cmap(img, 'viridis', 'example.png')
    """
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()

def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 5:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 6:
            y[index] = np.array([200, 100, 0]) / 255.
        if item == 7:
            y[index] = np.array([0, 200, 100]) / 255.
        if item == 8:
            y[index] = np.array([100, 0, 200]) / 255.
        if item == 9:
            y[index] = np.array([200, 0, 100]) / 255.
        if item == 10:
            y[index] = np.array([100, 200, 0]) / 255.
        if item == 11:
            y[index] = np.array([0, 100, 200]) / 255.
        if item == 12:
            y[index] = np.array([150, 75, 75]) / 255.
        if item == 13:
            y[index] = np.array([75, 150, 75]) / 255.
        if item == 14:
            y[index] = np.array([75, 75, 150]) / 255.
        if item == 15:
            y[index] = np.array([255, 100, 100]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([100, 100, 255]) / 255.
        if item == 18:
            y[index] = np.array([255, 150, 75]) / 255.
        if item == 19:
            y[index] = np.array([75, 255, 150]) / 255.
        if item == 20:
            y[index] = np.array([150, 75, 255]) / 255.
        if item == 21:
            y[index] = np.array([50, 50, 50]) / 255.
        if item == 22:
            y[index] = np.array([100, 100, 100]) / 255.
        if item == 23:
            y[index] = np.array([150, 150, 150]) / 255.
        if item == 24:
            y[index] = np.array([200, 200, 200]) / 255.
        if item == 25:
            y[index] = np.array([250, 250, 250]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y

    #     if item == 0:
    #         y[index] = np.array([255, 0, 0]) / 255.
    #     if item == 1:
    #         y[index] = np.array([0, 255, 0]) / 255.
    #     if item == 2:
    #         y[index] = np.array([0, 0, 255]) / 255.
    #     if item == 3:
    #         y[index] = np.array([255, 255, 0]) / 255.
    #     if item == 4:
    #         y[index] = np.array([0, 255, 255]) / 255.
    #     if item == 5:
    #         y[index] = np.array([255, 0, 255]) / 255.
    #     if item == 6:
    #         y[index] = np.array([192, 192, 192]) / 255.
    #     if item == 7:
    #         y[index] = np.array([128, 128, 128]) / 255.
    #     if item == 8:
    #         y[index] = np.array([128, 0, 0]) / 255.
    #     if item == 9:
    #         y[index] = np.array([128, 128, 0]) / 255.
    #     if item == 10:
    #         y[index] = np.array([0, 128, 0]) / 255.
    #     if item == 11:
    #         y[index] = np.array([128, 0, 128]) / 255.
    #     if item == 12:
    #         y[index] = np.array([0, 128, 128]) / 255.
    #     if item == 13:
    #         y[index] = np.array([0, 0, 128]) / 255.
    #     if item == 14:
    #         y[index] = np.array([255, 165, 0]) / 255.
    #     if item == 15:
    #         y[index] = np.array([255, 215, 0]) / 255.
    #     if item == 16:
    #         y[index] = np.array([0, 0, 0]) / 255.
    #     if item == 17:
    #         y[index] = np.array([215, 255, 0]) / 255.
    #     if item == 18:
    #         y[index] = np.array([0, 255, 215]) / 255.
    #     if item == -1:
    #         y[index] = np.array([0, 0, 0]) / 255.
    # return y



def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
    # print('y_train', np.unique(y_train))
    # print('y_val', np.unique(y_val))
    # print('y_test', np.unique(y_test))
    # print(y_val)
    # print(y_test)

    # K.clear_session()  # clear session before next loop

    # print(y1_train)
    #y1_train = to_categorical(y1_train)  # to one-hot labels
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return train_iter, valiada_iter, test_iter, all_iter #, y_test

def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices):
    pred_test = []
    # with torch.no_grad():
    #     for i in range(len(gt_hsi)):
    #         if i == 0:
    #             pred_test.extend([-1])
    #         else:
    #             X = all_iter[i].to(device)
    #             net.eval()  # 评估模式, 这会关闭dropout
    #             # print(net(X))
    #             pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

        # for X, y in all_iter:
        #     #for data, label in X, y:
        #     if y.item() != 0:
        #         # print(X)
        #         X = X.to(device)
        #         net.eval()  # 评估模式, 这会关闭dropout
        #         y_hat = net(X)
        #         # print(net(X))
        #         pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
        #     else:
        #         pred_test.extend([-1])
    for X, y in all_iter:
        X = X.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        # print(net(X))
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

#  修改的地方111111111111111111==============================
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            # x[i] = 16
            x_label[i] = 16
        # else:
        #     x_label[i] = pred_test[label_list]
        #     label_list += 1
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    # print('-------Save the result in mat format--------')
    # x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    # sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    timestamp = time.strftime("-%y-%m-%d-%H.%M")
    path = '/root/DBDA/' + net.name
    classification_map(y_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_' + net.name + timestamp + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + timestamp + '_gt.png')
    print('------Get classification maps successful-------')
