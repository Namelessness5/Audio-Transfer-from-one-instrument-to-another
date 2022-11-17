import numpy as np
from dataload import getdataset
from model import transfer
import torch
import os
import torch.nn as nn

convert_npy1 = "data/cn1/"
convert_npy2 = "data/cn2/"
convert_cqt1 = "data/cqt1/"
convert_cqt2 = "data/cqt2/"
model_path = "model/"
model_name = "bestmodelP-G"

cqt = False
method = "linear"
conv_dim = 8
model_depth = 4  # for conv: depth should be greater than 2
kernel_size = 19
bias = False

lr = 0.003
batch_size = 16
step_size = 50
loss_func = nn.MSELoss()
gamma = 0.85
n_fft = 512
epoch = 1000
nw = 4  # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])


def train(model, trainloader, epochs, criterion, optimizer, scheduler, device):
    model = model.to(device)
    minloss = np.inf

    for e in range(epochs):
        running_loss = 0
        for li, (inputs, labels) in enumerate(trainloader):
            if method == "conv" and not cqt:
                inputs = inputs.reshape((len(inputs), 1, n_fft // 2 + 1))
                labels = labels.reshape((len(inputs), 1, n_fft // 2 + 1))
            elif method == "conv" and cqt:
                inputs = inputs.reshape((len(inputs), 1, 84))
                labels = labels.reshape((len(inputs), 1, 84))
            inputs, labels = inputs.to(device), labels.to(device)

            # 前馈及反馈
            outputs = model(inputs)  # 数据前馈，正向传播
            loss = criterion(outputs, labels)  # 输出误差

            optimizer.zero_grad()  # 优化器梯度清零
            loss.backward()  # 误差反馈
            optimizer.step()  # 优化器更新参数

            running_loss += loss.item()

        scheduler.step()
        print("epoch:{}  loss:{}".format(e, running_loss))
        if running_loss < minloss:
            print(">>>>>>>>saving best model<<<<<<<<")
            torch.save(model.state_dict(), model_path + model_name + "_" + method + ".pth")
            minloss = running_loss
            print("DONE")


if __name__ == "__main__":
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if cqt:
        train_data = getdataset(convert_cqt1, convert_cqt2)
        model = transfer(84, depth=model_depth, method=method, conv_dim=conv_dim, kernel_size=kernel_size, bias=bias)
    else:
        train_data = getdataset(convert_npy1, convert_npy2)
        model = transfer(n_fft // 2 + 1, depth=model_depth, method=method, conv_dim=conv_dim, kernel_size=kernel_size,
                         bias=bias)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, epoch, loss_func, optimizer, scheduler, device)