import datetime as dt
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../')
import time
import cv2
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from libtiff import TIFF
from model_stage1 import *

start_time = dt.datetime.now().strftime('%F %T')
print("程序开始运行时间：" + start_time)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

##########################################################
#######                  网络超参数                 #######
##########################################################
w = 0.4
EPOCH = 40
BATCH_SIZE = 32
LR = 0.000193

print('EPOCH：', EPOCH)
print('BATCH_SIZE：', BATCH_SIZE)
print('LR：', LR)

##########################################################
#######              读取数据: 图片、标签           #######
##########################################################

ms_path = './dataset/xian_urban/ms4.tif'
pan_path = './dataset/xian_urban/pan.tif'
train_path = './dataset/xian_urban/train.npy'
test_path = './dataset/xian_urban/test.npy'
save_module_path = './dataset/xian_urban/model_stage1.pkl'

# # 读取图片、标签
ms4_tif = TIFF.open(ms_path, mode='r')
ms4_np = ms4_tif.read_image()
print('原始ms4图的形状：', np.shape(ms4_np))

pan_tif = TIFF.open(pan_path, mode='r')
pan_np = pan_tif.read_image()
print('原始pan图的形状;', np.shape(pan_np))

label_np = np.load(train_path)
# small_xian
label_np[label_np == 1] = 1
label_np[label_np == 2] = 2
label_np[label_np == 3] = 2
label_np[label_np == 4] = 2
label_np[label_np == 5] = 3
label_np[label_np == 6] = 4
label_np[label_np == 7] = 3

print('训练label数组形状：', np.shape(label_np))

test_label_np = np.load(test_path)
# small_xian
test_label_np[test_label_np == 1] = 1
test_label_np[test_label_np == 2] = 2
test_label_np[test_label_np == 3] = 2
test_label_np[test_label_np == 4] = 2
test_label_np[test_label_np == 5] = 3
test_label_np[test_label_np == 6] = 4
test_label_np[test_label_np == 7] = 3
print('测试label数组形状：', np.shape(test_label_np))

########################################################
#######                HS 与 SAR 补零            #######
########################################################

HS_patch_size = 16  # HS 截块的边长  16
Interpolation = cv2.BORDER_REFLECT_101

top_size, bottom_size, left_size, right_size = (int(HS_patch_size / 2 - 1), int(HS_patch_size / 2),
                                                int(HS_patch_size / 2 - 1), int(HS_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))

SAR_patch_size = HS_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(SAR_patch_size / 2 - 4), int(SAR_patch_size / 2),
                                                int(SAR_patch_size / 2 - 4), int(SAR_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))

####################################################
######            对训练数据label操作           ######
######       返回类别标签与各个类别所占的数量      ######
####################################################
gt_np = label_np - 1  # 标签中 0 类标签是未标注的像素，
#  通过减一后将类别归到 0-N，而未标注类标签变为255

label_element, element_count = np.unique(gt_np, return_counts=True)  # 去除数组中的重复数字，并将元素从小到大排序输出，即 0～8
Categories_Number = len(label_element) - 1  # 数据的类别数 9-1=8

print('类标：', label_element)  ### [0 1 2 3 4 5 6 7 255]
print('各类样本数：', element_count)  ###### [   443    423    499    376    331    280    298    170  817328]
print('标注的类别数：', Categories_Number)  #### 8类，其中第0类无标签，为255

####################################################
######            对测试数据label操作           ######
######       返回类别标签与各个类别所占的数量      ######
####################################################
test_gt_np = test_label_np - 1  # 标签中 0 类标签是未标注的像素，
#  通过减一后将类别归到 0-N，而未标注类标签变为255

test_label_element, test_element_count = np.unique(test_gt_np, return_counts=True)  # 去除数组中的重复数字，并将元素从小到大排序输出，即 0～8
test_Categories_Number = len(test_label_element) - 1  # 数据的类别数 9-1=8

print('测试类标：', test_label_element)  ### [0 1 2 3 4 5 6 7 255]
print('测试集各类样本数：', test_element_count)  ###### [ 54511 268219  19067  58906  17095  13025  24526   6502 358297]
print('测试标注的类别数：', test_Categories_Number)  #### 8类，其中第0类无标签，为255

#############################################################
######      统计 label 图中的 row、column 所有标签个数     ######
######                并统计每一类的个数                  ######
#############################################################
label_row, label_column = np.shape(gt_np)  # 获取 label 图的行、列: 1723,476
ground_xy = np.array([[]] * Categories_Number).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2)  # range:[0,820147]
ground_xy_allData = ground_xy_allData.reshape(label_row * label_column, 2)  # shape:(820148*2, ), size:1640296

count = 0
for row in range(label_row):  #### 行 1723
    for column in range(label_column):  #### 列 476
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if gt_np[row][column] != 255 and gt_np[row][column] != -1:
            ground_xy[int(gt_np[row][column])].append([row, column])

        ################################################################
######      统计 测试label 图中的 row、column 所有标签个数     ######
######                并统计每一类的个数                     ######
################################################################
test_label_row, test_label_column = np.shape(test_gt_np)  # 获取 label 图的行、列: 1723,476
test_ground_xy = np.array([[]] * test_Categories_Number).tolist()
test_ground_xy_allData = np.arange(test_label_row * test_label_column * 2)  # range:[0,820147]
test_ground_xy_allData = test_ground_xy_allData.reshape(test_label_row * test_label_column,
                                                        2)  # shape:(820148*2, ), size:1640296

test_count = 0
for test_row in range(test_label_row):  #### 行 1723
    for test_column in range(test_label_column):  #### 列 476
        test_ground_xy_allData[test_count] = [test_row, test_column]
        test_count = test_count + 1

        if test_gt_np[test_row][test_column] != 255 and test_gt_np[test_row][test_column] != -1:
            test_ground_xy[int(test_gt_np[test_row][test_column])].append([test_row, test_column])

        ######################################################
######                   标签内打乱               ######
######################################################
for categories in range(Categories_Number):  # 8类：从第0类开始循环，打乱标签
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)  # 三个参数 np.arange(a, b, c): 起点a，终点b，步长c
    np.random.shuffle(shuffle_array)
    ground_xy[categories] = ground_xy[categories][shuffle_array]  # 类别索引 + 位置

shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]  # shape:(820148*2, ), size:1640296

######################################################
######                 测试标签内打乱             ######
######################################################
for test_categories in range(test_Categories_Number):  # 8类：从第0类开始循环，打乱标签
    test_ground_xy[test_categories] = np.array(test_ground_xy[test_categories])
    test_shuffle_array = np.arange(0, len(test_ground_xy[test_categories]), 1)  # 三个参数 np.arange(a, b, c): 起点a，终点b，步长c
    np.random.shuffle(test_shuffle_array)
    test_ground_xy[test_categories] = test_ground_xy[test_categories][test_shuffle_array]  # 类别索引 + 位置

test_shuffle_array = np.arange(0, test_label_row * test_label_column, 1)
np.random.shuffle(test_shuffle_array)
test_ground_xy_allData = test_ground_xy_allData[test_shuffle_array]  # shape:(820148*2, ), size:1640296

#######################################################
######      从训练、测试的label中 按比例 选择数据      #####
######                  进行训练                   #####
######################################################
ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    for i in range(categories_number):
        ground_xy_train.append(ground_xy[categories][i])

    label_train = label_train + [categories for x in range(int(categories_number))]

for test_categories in range(test_Categories_Number):
    test_categories_number = len(test_ground_xy[test_categories])
    # print('aaa', categories_number)
    for i in range(test_categories_number):
        ground_xy_test.append(test_ground_xy[test_categories][i])

    label_test = label_test + [test_categories for x in range(int(test_categories_number))]

label_train = np.array(label_train)  # shape:(2280, ), size: 2280  训练集label图片位置
label_test = np.array(label_test)  # shape:(0, ), size: 0
ground_xy_train = np.array(ground_xy_train)  # shape:(2280, 2), size: 5640 训练集的图片位置
ground_xy_test = np.array(ground_xy_test)  # shape:(0, ), size: 0

#############################################
######         训练数据、测试数据         ######
######            数据集内打乱           ######
#############################################
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('训练样本数：', len(label_train))  #### 训练样本数： 2820
print('测试样本数：', len(label_test))  ####  测试样本数： 461851


###############################
######   归一化 图片函数   ######
###############################
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


####################################
#####   HS 和 SAR 数据进行归一化  #####
#####    转换成 Float 数据类型    #####
####################################
HS = to_tensor(ms4_np)
SAR = to_tensor(pan_np)

SAR = np.expand_dims(SAR, axis=0)
HS = np.array(HS).transpose((2, 0, 1))  # 调整通道 H，W，C --> C，H，W

HS = torch.from_numpy(HS).type(torch.FloatTensor)
SAR = torch.from_numpy(SAR).type(torch.FloatTensor)

class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        # def __init__(self, MS4, Pan, MSHPAN, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        # self.train_data3 = MSHPAN
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]


        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)
class MyData1(Dataset):
    # def __init__(self, MS4, Pan, MSHPAN, xy, cut_size):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        # self.train_data3 = MSHPAN
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]


        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)

train_data = MyData(HS, SAR, label_train, ground_xy_train, HS_patch_size)  # 先对数据作切片
test_data = MyData(HS, SAR, label_test, ground_xy_test, HS_patch_size)  # 更改HS, SAR, label_test, ground_xy_test所有操作
all_data = MyData1(HS, SAR, ground_xy_allData, HS_patch_size)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE * 20, shuffle=False, num_workers=0)
all_data_loader = Data.DataLoader(dataset=all_data, batch_size=BATCH_SIZE * 20, shuffle=False, num_workers=0)
# test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE *  20, shuffle=False, num_workers=0)
# all_data_loader = Data.DataLoader(dataset=all_data, batch_size=BATCH_SIZE *  20, shuffle=False, num_workers=0)

cnn = Net(4, 1, Categories_Number)


# ================#
#     GPU并行    #
# ===============#
# if torch.cuda.device_count() > 1:
#     cnn = nn.DataParallel(cnn, device_ids=[0,1,2]).cuda()


# ===============#
#   调整学习率    #
# ===============#
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.9995


# ===================#
#   参数初始化方法    #
# ===================#

for m in Net.modules(cnn):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

# ==============================#
#   使用cuda，优化器，loss函数    #
# ==============================#
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.0005)  # optimize all cnn parameters
loss_func0 = nn.CrossEntropyLoss()  # the target label is not one-hotted
loss_func1 = nn.CrossEntropyLoss()  # the target label is not one-hotted
loss_func2 = nn.CrossEntropyLoss()  # the target label is not one-hotted


# ===================#
#      开始训练      #
# ===================#
trainstart = time.time()
best_acc = 0
print("===========   let's training    ===========")
for epoch in range(EPOCH):
    valid_batch = iter(test_loader)
    for step, (HS, SAR, label, _) in enumerate(train_loader):
        cnn.train()
        HS = HS.cuda()
        SAR = SAR.cuda()

        label = label.cuda()

        output, _, _ = cnn(HS, SAR)
        loss = loss_func0(output, label)

        optimizer.zero_grad()
        loss.backward()

        # 获取梯度
        grad = []
        for param in cnn.parameters():
            if param.grad is not None:
                grad.append(param.grad.norm().item())
        gr = torch.tensor(grad).mean().item()

        optimizer.step()

        cnn.eval()
        if step % 100 == 0:
            HS_test1, SAR_test1, label_test1, _ = next(valid_batch)
            HS_test1 = HS_test1.cuda()
            SAR_test1 = SAR_test1.cuda()

            label_test1 = label_test1.cuda()

            with torch.no_grad():
                test_output, _, _ = cnn(HS_test1, SAR_test1)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()


            accuracy = (pred_y == label_test1).sum().item() / float(label_test1.size(0))
            print('Epoch:%3d' % epoch, '|| step: %3d' % step, '|| train loss: %.4f' % loss.item(), '|| grad: %.4f' % gr,
                  '|| test acc: %.4f' % accuracy)


        adjust_learning_rate(optimizer, epoch)

    if epoch > 10:
        a = 0
        y_pred_ = []
        for step, (HS, SAR, label, gt_xy) in enumerate(test_loader):
            a = a + 1
            HS = HS.cuda()
            SAR = SAR.cuda()
            label = label.cuda()
            with torch.no_grad():
                output, _, _ = cnn(HS, SAR)  # cnn output

            pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
            if a == 1:
                y_pred_ = pred_y.cpu().numpy()
            else:
                y_pred_ = np.concatenate((y_pred_, pred_y.cpu().numpy()), axis=0)
        con_mat = confusion_matrix(y_true=label_test, y_pred=y_pred_)
        OA = np.trace(con_mat) / np.sum(con_mat)
        print('OA:', OA)

        AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))  # axis=1 每行求和
        print('AA:', AA)

        Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
        Kappa = (OA - Pc) / (1 - Pc)
        print('Kappa:', Kappa)

        if OA > best_acc:
            best_acc = OA
            torch.save(cnn, save_module_path)
            print("更新模型")


trainend = time.time()
# 保存模型

cnn2 = torch.load(save_module_path)
cnn2.cuda()

l = 0
y_pred = []
cnn2.eval()
class_count = np.zeros(7)
for step, (HS, SAR, label, gt_xy) in enumerate(test_loader):
    l = l + 1
    HS = HS.cuda()
    SAR = SAR.cuda()

    label = label.cuda()
    with torch.no_grad():

        output, _, _ = cnn2(HS, SAR)

    pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
    if l == 1:
        y_pred = pred_y.cpu().numpy()
    else:
        y_pred = np.concatenate((y_pred, pred_y.cpu().numpy()), axis=0)

testend = time.time()
traintime = trainend - trainstart
testtime = testend - trainend
print("train time: %.2f S || test time: %.2f S" % (traintime, testtime))
con_mat = confusion_matrix(y_true=label_test, y_pred=y_pred)
print('con_mat', con_mat)

# ===================#
#     计算性能参数    #
# ===================#
all_acr = 0
p = 0
column = np.sum(con_mat, axis=0)  # 列求和
line = np.sum(con_mat, axis=1)  # 行求和
for i, clas in enumerate(con_mat):
    precise = clas[i]
    all_acr = precise + all_acr
    acr = precise / column[i]
    recall = precise / line[i]
    f1 = 2 * acr * recall / (acr + recall)
    temp = column[i] * line[i]
    p = p + temp
    print("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f " % (i, acr, recall, f1))
OA = np.trace(con_mat) / np.sum(con_mat)
print('OA:', OA)

AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))  # axis=1 每行求和
print('AA:', AA)

Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
Kappa = (OA - Pc) / (1 - Pc)
print('Kappa:', Kappa)



end_time = dt.datetime.now().strftime('%F %T')
print("程序结束运行时间：" + end_time)


