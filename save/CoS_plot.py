import matplotlib.pyplot as plt
import numpy as np

# 生成一组随机数据

filename_acc = 'D:\PythonCode\FCFL\my_fcfl\save\select_list_20.txt'
data = []
data_int = []
# 相比open(),with open()不用手动调用close()方法
with open(filename_acc, 'r') as f:
    lines = f.readlines() 
    for line in lines:
        value = line.split()
        data.append(value)
for i in data:
    data_int.extend(int(value) for value in i)

data_list = []
# print(data_int)
for i in data_int:
    data_list.append([i])

# print(data_list)


data_samplesize = data_int[0:20]
data_random = data_int[20:40]
data_unfairness = data_int[40:]

# print(data_unfairness)


bar_width = 0.3
x = np.arange(20) 
# 绘图 x 表示 从哪里开始
plt.figure(figsize=(11,9.5))
plt.bar(x, data_unfairness, bar_width, label='unfairness_select', hatch="//")
plt.bar(x+bar_width, data_random, bar_width, align="center", label='random_select', hatch="")
plt.bar(x+bar_width+bar_width, data_samplesize, bar_width, align="center", label='data_amount_select', hatch="-") 


# 绘制直方图
# plt.hist(data_unfairness, range=[0,29], bins=np.arange(-0.5, 30 + 1.5, 1), label='unfairness_select')
# plt.hist(data_random, bins=30, alpha=0.5, label='random_select')
# plt.hist(data_samplesize, bins=30, alpha=0.5, label='sample_size_select')

plt.xticks(np.arange(20), ["%d" %c_id for c_id in range(20)], fontsize = "20")
plt.yticks(fontsize = "20")
# 设置图表属性
# plt.title('Number of selection')
plt.xlabel('Client_ID', fontsize = "20")
plt.ylabel('Number of selection', fontsize = "20")
plt.legend(fontsize = "20", loc = 'upper right')

# 显示图表
plt.show()