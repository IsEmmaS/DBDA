import time
import torch
import numpy as np
import sys
from pathlib import Path
PWD = Path(__file__).resolve().parent.parent
sys.path.append(f'{PWD}/global_module/')
import d2lzh_pytorch as d2l

def evaluate_accuracy(data_iter, net, loss, device):
    """
    评估模型在给定数据集上的准确率和损失。

    Args:
        data_iter (DataLoader): 数据迭代器，用于加载评估数据。
        net (nn.Module): 待评估的神经网络模型。
        loss (nn.Module): 损失函数，用于计算模型输出与真实标签之间的损失。
        device (str): 设备类型，如'cuda'或'cpu'，指定模型和数据运行的设备。

    Returns:
        list: 包含准确率和平均损失的列表。
    """
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = X.to(device)
            y = y.to(device)
            net.eval() # 评估模式, 这会关闭dropout
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() # 改回训练模式
            n += y.shape[0]
    return [acc_sum / n, test_l_sum] # / test_num]

def train(net, train_iter, valida_iter, loss, optimizer, device, datasets,image_path,iter_index,epochs=30, early_stopping=True,early_num=20,):
    """
    训练神经网络模型。

    Args:
        net (nn.Module): 待训练的神经网络模型。
        train_iter (DataLoader): 训练数据迭代器。
        valida_iter (DataLoader): 验证数据迭代器。
        loss (nn.Module): 损失函数。
        optimizer (nn.Module): 优化器，用于更新模型参数。
        device (str): 设备类型，指定模型和数据运行的设备。
        datasets (str): 数据集名称，用于保存图像时标识。
        image_path (str): 保存训练图像的路径。
        iter_index (int): 当前迭代次数。
        epochs (int, optional): 训练轮数，默认为30。
        early_stopping (bool, optional): 是否启用早停，默认为True。
        early_num (int, optional): 早停轮数，默认为20。

    Returns:
        None
    """
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1) # 在每个epoch外创建lr_adjust
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step()# epoch) # 新特性将删除epoch参数，移除避免警告
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        # 转换到cpu上运行numpy
        valida_loss_list.append(valida_loss.cpu().numpy())
        # loss_list.append(valida_loss)
        loss_list.append(valida_loss.cpu().numpy())

        # 绘图部分
        train_loss_list.append(train_l_sum) # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        # valida_loss_list.append(valida_loss)
        valida_loss_list.append(valida_loss.cpu().numpy())
        valida_acc_list.append(valida_acc)

        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss.cpu().item(), valida_acc, time.time() - time_epoch))

        PATH = "./net_DBA.pt"
        # if loss_list[-1] <= 0.01 and valida_acc >= 0.95:
        #     torch.save(net.state_dict(), PATH)
        #     break

        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            if early_epoch == 0: # and valida_acc > 0.9:
                torch.save(net.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0

    d2l.set_figsize()
    d2l.plt.figure(figsize=(8, 8.5))
    train_accuracy = d2l.plt.subplot(221)
    train_accuracy.set_title('train_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train_accuracy')
    # train_acc_plot = np.array(train_acc_plot)
    # for x, y in zip(num_epochs, train_acc_plot):
    #    d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    test_accuracy = d2l.plt.subplot(222)
    test_accuracy.set_title('valida_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('test_accuracy')
    # test_acc_plot = np.array(test_acc_plot)
    # for x, y in zip(num_epochs, test_acc_plot):
    #   d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    loss_sum = d2l.plt.subplot(223)
    loss_sum.set_title('train_loss')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='red')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train loss')
    # ls_plot = np.array(ls_plot)

    test_loss = d2l.plt.subplot(224)
    test_loss.set_title('valida_loss')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('valida loss')
    # ls_plot = np.array(ls_plot)

    # d2l.plt.show()
    d2l.plt.tight_layout()
    date = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    d2l.plt.savefig(f'{image_path}/{iter_index}-{datasets}-{date}.png')
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec, figures @%s'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start, image_path))
