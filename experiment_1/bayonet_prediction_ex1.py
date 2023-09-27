# coding=utf8
import numpy as np
import time
result_list = []

"""
data = ['"21st & I St NW"', '"Adams Mill & Columbia Rd NW"', '"21st & I St NW"', '"Adams Mill & Columbia Rd NW"',
            '"Massachusetts Ave & Dupont Circle NW"']
data = ['"37th & O St NW / Georgetown University"', '"Massachusetts Ave & Dupont Circle NW"']
"""


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


class new_node:
    def __init__(self):
        # key是名字 child是子节点，support是支持度
        self.key = ""
        self.support = 1


def main():
    trajectory_list_list = []
    trajectory_list = []
    # test的路径
    train_bayonet_csv_file = "./real_data/train_test_data_4/train_data_3.csv"
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
                # test_trajectory_list.append(str(test_bayonet_data[i][1]))

    tree = T_tree("root")
    # 遍历搜索所有人的轨迹
    # print(len(tree.child))
    # time.sleep(100)
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
    # print(len(tree.child), len(tree.child[0].child))
    # time.sleep(1000)
    return tree
    # ---------------------------------------上方构建了统计学习的模型T-tree，下面进行预测-------------------------------------------------



def t_tree_dfs_prediction(tmp, node, data):
    # print(data)
    # time.sleep(0.1)
    # 先分两类 第一在已经找到所有的节点，找子节点进行下一步预测
    if len(tmp) == len(data):
        # print("11111111")
        # 判断一下这个节点有没有子节点，如果有子节点，加入结果集合中
        if node.child:
            for next_node in node.child:
                # print(node.child)
                # time.sleep(3)
                global result_list
                result = []
                result.extend(tmp)
                # print(result)
                # time.sleep(100)
                result.append(next_node)
                result_list.append(result)
                # print(next_node.key,score)
            tmp = []
            return tmp
        # 如果没有子节点，就是说明这里找不到预测的节点，就直接返回，无需加入在结果集中
        else:
            # print(node)
            tmp = []
            return tmp

    # 第二类目前还没这样的长度，那么就继续找
    if len(tmp) < len(data):
        if node.child:
            # 依次遍历子节点，其实就是DFS
            for next_node in node.child:
                # print()
                # print(tmp, node.key)
                # 如果找到了，就加入结果集合中
                if next_node.key == data[len(tmp)]:
                    # print(tmp)
                    # time.sleep(0.000000001)
                    tmp.append(next_node)
                    tmp_node = next_node
                    tmp = t_tree_dfs_prediction(tmp, tmp_node, data)
            # 如果flag（找到结果了）和tmp（暂时存储的集合中存在节点），那么就返回集合元素减一
            tmp = []
            for next_node in node.child:
                tmp = t_tree_dfs_prediction(tmp, next_node, data)
            # if flag and tmp:
            #    # print(ret_next_node)
            #    tmp = tmp[:-1]
        else:
            tmp = []
            return tmp
    tmp = []
    return tmp


def predict(tree, data):
    global result_list
    result_list = []
    # 定义五个变量 也是函数中要遇到的
    # tmp就是目前的长度，进行分类需要用
    tmp = []
    # 初始化节点，tree是当时建立的结构体，node就是tree
    node = tree
    # 深度优先遍历搜索所有节点，对找到支持度最大的节点进行输出
    # 此dfs遍历时记得加了几次就要减几次
    t_tree_dfs_prediction(tmp, node, data)
    # print(result_list)
    # 整合数据 去重
    lst1 = []
    for i in result_list:
        if i not in lst1:
            lst1.append(i)
    # print(lst1)
    result_list = lst1
    # print(len(result_list))
    # time.sleep(100)
    # 最终结果集合
    get_list = []
    tmp_get_list = []
    # 对数据进行整合，如果一直，那就支持度叠加
    for each_list in result_list:
        # 如果结果列表存在数据继续进行搜搜索
        if get_list:
            # 依次遍历最终列表的元素
            # flag查看能不能找到一致的
            flag = 0
            for index_get_list, each_get_list in enumerate(get_list):
                # 每次到新列表级数器就计0
                count = 0
                # 与原来列表的每个节点对比，如果一样，那就对比下一个，然后计数器+1
                for index, each_point in enumerate(each_list, 0):
                    # 已经找到了
                    if flag == 1:
                        break
                    # 如果一致，计数器加一，否则看下一个列表
                    if each_point.key == each_get_list[index].key:
                        count += 1
                    else:
                        break
                    # 如果数量相等,找到了，flag置1,对这些元素依次+1
                    if count == len(data) + 1:
                        flag = 1

                        for i in range(len(data) + 1):
                            # print(get_list[index_get_list][i].support)
                            # node = get_list[index_get_list][i]
                            # print(node.support)
                            new_node_tmp = new_node()
                            new_node_tmp.support = each_list[i].support + get_list[index_get_list][i].support
                            new_node_tmp.key = get_list[index_get_list][i].key
                            # print(node.support,node)
                            # print(new_node_tmp.support,new_node_tmp.key)
                            # time.sleep(1)
                            get_list[index_get_list][i] = new_node_tmp
            # 如果最后没找到，那就把当前的放入结果列表中
            if flag == 0:
                get_list.append(each_list)
        # 如果不存在就添加列表
        else:
            get_list.append(each_list)


    # 这部分就是计算哪组数据是分数最高的
    tmp_name = []
    name = []
    result_name = []
    highest_score = 0
    for i in get_list:
        # print(i)
        sum = 0
        for each in i[-1:]:
            sum += each.support
            tmp_name.append(each.key)
            # print(tmp_name)
        name.append(tmp_name)
        # print("-----------")
        # print(sum,each.key)
        if sum > highest_score:
            # print(sum)
            highest_score = sum
            result_name = each.key

    # print(name)
    # print("-------------------------------------------------------------")
    # print("最终预测结果： " + result_name + "最高支持度： " + str(highest_score))
    return result_name




# 对于真实卡口轨迹的处理
def produce_test_data():
    test_trajectory_list_list = []
    test_trajectory_list = []
    # test的路径
    test_bayonet_csv_file = "./real_data/train_test_data_4/test_data_3.csv"
    test_bayonet_data = np.loadtxt(test_bayonet_csv_file, dtype="str", delimiter=",", skiprows=1, usecols=[1, 2])
    # print(type(str(test_bayonet_data[0][0])),str(test_bayonet_data[0][0]))
    # time.sleep(1000)
    for i in range(len(test_bayonet_data)):
        if not test_trajectory_list:
            test_trajectory_list.append(str(test_bayonet_data[i][0]))
            # test_trajectory_list.append(str(test_bayonet_data[i][1]))
            continue
        if test_trajectory_list:
            if (test_bayonet_data[i - 1][1] == test_bayonet_data[i][1]):
                # test_trajectory_list.append("\""+str(test_bayonet_data[i][0])+"\"")
                test_trajectory_list.append(str(test_bayonet_data[i][0]))
            else:
                test_trajectory_list_list.append(test_trajectory_list)
                test_trajectory_list = []
                test_trajectory_list.append(str(test_bayonet_data[i][0]))
                # test_trajectory_list.append(str(test_bayonet_data[i][1]))
                # print(test_trajectory_list_list)
                # time.sleep(5)
    # print(test_trajectory_list_list[:10])
    # time.sleep(10000)
    return test_trajectory_list_list
    
if __name__ == '__main__':
    true = 0
    false = 0
    test_trajectory_data = produce_test_data()
    tree = main()
    for data in test_trajectory_data:
        result = ""
        # if len(data) != 4:
        #     data = data[:4]
        # print(data[:2])
        try:
            while result == "" or result == []:
                result = predict(tree, data[:-1])
                print(data)
                data = data[1:]
            if result == data[-1]:
                # print(data[:2])
                # print(result,data[2])
                # print("预测成功")
                # time.sleep(5)
                true += 1
            else:
                # print(data[:2])
                # print(result,data[2])
                # print("预测失败")
                # time.sleep(5)
                false += 1
        except:
            # print(data)
            # time.sleep(100)
            false += 1
        print(true,false)
