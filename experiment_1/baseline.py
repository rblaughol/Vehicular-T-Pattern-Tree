# coding=utf8
import numpy as np
import time


trajectory_list_list = []
trajectory_list = []
# test的路径
train_bayonet_csv_file = "./real_data/train_test_data_3/train_data_3.csv"
train_bayonet_data = np.loadtxt(train_bayonet_csv_file, dtype="str", delimiter=",", skiprows=1, usecols=[1, 2])
# print(type(str(test_bayonet_data[0][0])),str(test_bayonet_data[0][0]))
# time.sleep(1000)
for i in range(len(train_bayonet_data)):
    if not trajectory_list:
        trajectory_list.append(str(train_bayonet_data[i][0]))
        # test_trajectory_list.append(str(test_bayonet_data[i][1]))
        continue
    if trajectory_list:
        if (train_bayonet_data[i - 1][1] == train_bayonet_data[i][1]):
            # test_trajectory_list.append("\""+str(test_bayonet_data[i][0])+"\"")
            trajectory_list.append(str(train_bayonet_data[i][0]))
        else:
            trajectory_list_list.append(trajectory_list)
            trajectory_list = []
            trajectory_list.append(str(train_bayonet_data[i][0]))
print(trajectory_list_list[:10])
print("\n")


# 生成了这样的轨迹数据以后，然后就可以构造树了。
# ===========================================上方对轨迹数据进行了预处理，下面构建统计学习模型T-tree===========================================================

# 构造一个T_tree
class T_tree:
    def __init__(self, name):
        # key是名字 child是子节点，support是支持度
        self.key = name
        self.child = []
        self.support = 1

    # 获取节点，传入名字，返回这个对象
    def get(self, getname):
        # 遍历每一个子节点
        for each_child in self.child:
            # 如果找到名字一样的
            if each_child.key == getname:
                # 那就准备返回这个节点
                child_node = each_child
                break
        return child_node

    # 插入节点，如果节点存在，就支持度加一，否则就在子节点列表中添加这个节点
    def insert(self, newname):
        # 这里和get是一样的，只不过就是support+1
        if self.child:
            for each_child in self.child:
                if each_child.key == newname:
                    each_child.support += 1
                    return 0
        # 如果没找到，那就新建一个对象，然后把对象放在子节点列表中
        node = T_tree(newname)
        self.child.append(node)
        return 0


# 建立root节点
tree = T_tree("root")
# 遍历搜索所有人的轨迹

for each_indivisual_list in trajectory_list_list:
    # 遍历每个节点
    # print(tree.child, each_indivisual_list)
    for index, point in enumerate(each_indivisual_list, 0):
        # 节点数一定时两个或以上，但是，index=0和index>0分类讨论
        # 如果index=0，直接插入进去
        # print(index)
        if index == 0:
            # print(point)
            tree.insert(point)
        # 不是第一个节点，那就得继续找原来路径
        else:
            # 当前节点时root
            current_node = tree
            # print(tree.child, index)
            # 如果时第二个节点（不算root），那就先找第一个节点
            for i in range(index):
                # print(i)
                # 比如第一次就找each_indivisual_list[i]，第0个节点的名字，get到，然后在找第二个节点。。。找完一会加入最后一个节点
                current_node = current_node.get(each_indivisual_list[i])
            current_node.insert(point)

# for i in tree.child:
#     print(i.key)
#     print(i.support)
#
# for i in tree.child[0].child:
#     print(i.key)
#     print(i.support)
# ---------------------------------------上方构建了统计学习的模型T-tree，下面进行预测-------------------------------------------------
# print(tree.get('"M St & New Jersey Ave SE"'))
# tree.get("M St & New Jersey Ave SE")
# #data = ['"21st & I St NW"', '"Adams Mill & Columbia Rd NW"', '"21st & I St NW"', '"Adams Mill & Columbia Rd NW"',
# #        '"Massachusetts Ave & Dupont Circle NW"']
# #data = ['"7th & T St NW"', '"15th & P St NW"']
# # for i in tree.child:
# #     if i ==
#data_list = []
test_trajectory_list_list = []
test_trajectory_list = []
# test的路径
test_bayonet_csv_file = "./real_data/train_test_data_3/test_data_3.csv"
test_bayonet_data = np.loadtxt(test_bayonet_csv_file, dtype="str", delimiter=",", skiprows=1, usecols=[1, 2])
# print(type(str(test_bayonet_data[0][0])),str(test_bayonet_data[0][0]))
# time.sleep(1000)
for i in range(len(test_bayonet_data)):
    if not test_trajectory_list:
        test_trajectory_list.append(str(test_bayonet_data[i][0]))
        # test_trajectory_list.append(str(test_bayonet_data[i][1]))
        continue
    if test_trajectory_list:
        if test_bayonet_data[i - 1][1] == test_bayonet_data[i][1]:
            # test_trajectory_list.append("\""+str(test_bayonet_data[i][0])+"\"")
            test_trajectory_list.append(str(test_bayonet_data[i][0]))
        else:
            test_trajectory_list_list.append(test_trajectory_list)
            test_trajectory_list = [str(test_bayonet_data[i][0])]

data_list = test_trajectory_list_list
# print(len(data_list))
# time.sleep(100)
true = 0
false = 0
# print(data_list)
# print(len(tree.child))
data_l = 0
for data in data_list:
    # if data_l < len(data):
    #     data_l = len(data)
    # if len(data) != 4:
    #     data = data[:4]
    score = 0
    tmp_tree = tree
    try:
        for data_point in data[:-1]:
            tmp_tree = tmp_tree.get(data_point)
        # print(tree.child)
        score = 0
        for child_point in tmp_tree.child:
            if child_point.support > score:
                score = child_point.support
                name = child_point.key

        #print("最高支持度：",score)
        #print("预测的最终节点是：" + name)
        if name == data[-1]:
            true += 1
        else:
            false += 1
    except:
        try:
            for data_point in data[:-2]:
                tmp_tree = tmp_tree.get(data_point)
                # print(tree.child)
            score = 0
            for child_point in tmp_tree.child:
                if child_point.support > score:
                    score = child_point.support
                    name = child_point.key

            #print("最高支持度：",score)
            #print("预测的最终节点是：" + name)
            if name == data[-1]:
                true += 1
            else:
                false += 1
        except:
            false += 1
        #print(data)

    print(true, false)
#print(data_l)

