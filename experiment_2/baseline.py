# coding=utf8
import numpy as np
import time

# 读取数据
# bayonet_data_csv_file = "/Users/rb/Desktop/python_work/test_work_space/bayonet_data.csv"
bayonet_data_csv_file = "./experiment_data/new_data/train_3.csv"
bayonet_data = np.loadtxt(bayonet_data_csv_file, dtype="str", delimiter=",", skiprows=1, usecols=(1, 4, 6, 7))

# 存在一些有问题的数据，筛选掉
delete = []
for index, id in enumerate(bayonet_data[:, 3]):
    # 乱码的长度不为6，直接找到对应索引，直接删除就可以了
    #print(id,len(id))
    if len(id) != 6:
        delete.append(index)
bayonet_data = np.delete(bayonet_data, delete, axis=0)
# print(bayonet_data)

# 先找到人员的列表,去重
id_list = np.unique(bayonet_data[:, 3])
# print(id_list)

# 轨迹列表：
trajectory_list = []
for id in id_list:
    # 这里先找到属于id（人员）的轨迹
    indivisual_data = bayonet_data[id == bayonet_data[:, 3]]
    # 按照时间戳进行排序，sort是排序结果，argsort是排序后结果在原来多少行
    indivisual_data = indivisual_data[indivisual_data[:, 0].argsort()]
    # print(indivisual_data)
    # 索引值
    i = 0
    # 临时的轨迹数据，存储一个起始节点
    trajectory_list_tem = [indivisual_data[i, 1]]
    # 如果还有数据就一直循环,加一是防止i + 1报错
    while (len(indivisual_data) > i + 1):
        # 每次找两个数据，判断两个数据是否能连接
        # 如果两个数据能连接，那么存储在临时轨迹列表中
        # time.sleep(0.1)
        # print(indivisual_data[i],indivisual_data[i + 1])
        if indivisual_data[i, 2] == indivisual_data[i + 1, 1]:
            # 存储起来，然后索引加一，继续看下一个能不能连起来
            trajectory_list_tem.append(indivisual_data[i, 2])
            i = i + 1
        # 如果两个数据不能连接，那么连起来，临时轨迹列表清空，索引+1，最终结果加入这个轨迹序列
        elif indivisual_data[i, 2] != indivisual_data[i + 1, 1]:
            # 在结果中加入轨迹序列
            # print(indivisual_data)
            trajectory_list_tem.append(indivisual_data[i, 2])
            trajectory_list.append(trajectory_list_tem)
            # 重新找到起点
            trajectory_list_tem = [indivisual_data[i + 1, 1]]
            i = i + 1
            # time.sleep(5)
            # print(trajectory_list)
    # 加一了就会少一个没进去，这里收尾
    trajectory_list_tem.append(indivisual_data[i, 2])
    trajectory_list.append(trajectory_list_tem)

# print(trajectory_list[:10])
# print("\n")


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

for each_indivisual_list in trajectory_list:
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
# data = ['"21st & I St NW"', '"Adams Mill & Columbia Rd NW"', '"21st & I St NW"', '"Adams Mill & Columbia Rd NW"',
#         '"Massachusetts Ave & Dupont Circle NW"']
# data = ['"7th & T St NW"', '"15th & P St NW"']
# for i in tree.child:
#     if i ==
test_trajectory_list_list = []
test_trajectory_list = []
# test的路径
bayonet_csv_file = "./experiment_data/new_data/test_3.csv"
bayonet_data = np.loadtxt(bayonet_csv_file, dtype="str", delimiter=",", skiprows=1, usecols=[1,4,6,7])
# time.sleep(1000)
# 存在一些有问题的数据，筛选掉
delete = []
for index, id in enumerate(bayonet_data[:, 3]):
    # 乱码的长度不为6，直接找到对应索引，直接删除就可以了
    #print(id,len(id))
    if len(id) != 6:
        delete.append(index)
bayonet_data = np.delete(bayonet_data, delete, axis=0)
# print(bayonet_data)

# 先找到人员的列表,去重
id_list = np.unique(bayonet_data[:, 3])
# print(id_list)

# 轨迹列表：
trajectory_list = []
for id in id_list:
    # 这里先找到属于id（人员）的轨迹
    indivisual_data = bayonet_data[id == bayonet_data[:, 3]]
    # 按照时间戳进行排序，sort是排序结果，argsort是排序后结果在原来多少行
    indivisual_data = indivisual_data[indivisual_data[:, 0].argsort()]
    # print(indivisual_data)
    # 索引值
    i = 0
    # 临时的轨迹数据，存储一个起始节点
    trajectory_list_tem = [indivisual_data[i, 1]]
    # 如果还有数据就一直循环,加一是防止i + 1报错
    while (len(indivisual_data) > i + 1):
        # 每次找两个数据，判断两个数据是否能连接
        # 如果两个数据能连接，那么存储在临时轨迹列表中
        # time.sleep(0.1)
        # print(indivisual_data[i],indivisual_data[i + 1])
        if indivisual_data[i, 2] == indivisual_data[i + 1, 1]:
            # 存储起来，然后索引加一，继续看下一个能不能连起来
            trajectory_list_tem.append(indivisual_data[i, 2])
            i = i + 1
        # 如果两个数据不能连接，那么连起来，临时轨迹列表清空，索引+1，最终结果加入这个轨迹序列
        elif indivisual_data[i, 2] != indivisual_data[i + 1, 1]:
            # 在结果中加入轨迹序列
            # print(indivisual_data)
            trajectory_list_tem.append(indivisual_data[i, 2])
            trajectory_list.append(trajectory_list_tem)
            # 重新找到起点
            trajectory_list_tem = [indivisual_data[i + 1, 1]]
            i = i + 1
            # time.sleep(5)
            # print(trajectory_list)
    # 加一了就会少一个没进去，这里收尾
    trajectory_list_tem.append(indivisual_data[i, 2])
    trajectory_list.append(trajectory_list_tem)

"""
for i in range(len(test_bayonet_data)):
    if not test_trajectory_list:
        test_trajectory_list.append(str(test_bayonet_data[i][0]))
        test_trajectory_list.append(str(test_bayonet_data[i][1]))
        continue
    if test_trajectory_list:
        if (test_bayonet_data[i - 1][1] == test_bayonet_data[i][0]) and (
                test_bayonet_data[i - 1][2] == test_bayonet_data[i][2]):
            # test_trajectory_list.append("\""+str(test_bayonet_data[i][0])+"\"")
            test_trajectory_list.append(str(test_bayonet_data[i][1]))
        else:
            test_trajectory_list_list.append(test_trajectory_list)
            test_trajectory_list = []
            test_trajectory_list.append(str(test_bayonet_data[i][0]))
            test_trajectory_list.append(str(test_bayonet_data[i][1]))
"""
true = 0
false = 0
# print(data_list)
# print(len(tree.child))
# print(len(tree.child))
data_l = 0
for data in trajectory_list:
    # if data_l < len(data):
    #     data_l = len(data)
    # if len(data) != 4:
    #     data = data[:4]
    #print(data)
    score = 0
    tmp_tree = tree
    # te = tree.get("20th & E St NW")
    # for i in te.child:
    #     print(i.support)
    #     print(i.key)
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

            # print("最高支持度：",score)
            # print("预测的最终节点是：" + name)
            if name == data[-1]:
                true += 1
            else:
                false += 1
        except:
            false += 1
        #print(data)

    print(true, false)
