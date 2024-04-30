import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
import json
import numpy as np

from tools import remove_special_chars, has_word, VQAEval

tiankong = VQAEval()

def yes_or_no(answer, response):
    if isinstance(response, int):
        pass
    else:
        return tiankong.evaluate(response, answer)

# x 划分的区间
x_bins = [1000, 2000, 3000, 5000, 9000]
# y 的划分档位
y_interval = 0.1

# 初始化二维数组，x 划分为5档，y 划分为10档（因为 0-1 划分为0.1的10个区间）
total = np.zeros((len(x_bins) + 1, int(1 / y_interval)))
correct = np.zeros((len(x_bins) + 1, int(1 / y_interval)))

# jsonl 文件路径
jsonl_file_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer/it.jsonl'

# 读取 jsonl 文件
with open(jsonl_file_path, 'r') as file:
    for line in file:
        entry = json.loads(line)
        x = entry['token_max']
        y = entry['needle_location']
        z = entry['text']
        answer = entry['answer']

        # 确定 x 的档位
        x_index = np.digitize(x, x_bins)
        # 确定 y 的档位
        y_index = int(y / y_interval)

        # 将 z 值加到对应的档位中
        total[x_index][y_index] += 1
        if yes_or_no(answer, z):
            correct[x_index][y_index] += 1
            
    result = np.divide(correct, total, out=np.zeros_like(correct), where=total!=0)

# 打印结果
        

# 这里可以输出 data 数组查看结果或进一步处理
print(result)

# # Plot a heatmap for a numpy array:

file_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/heatmap.png'
uniform_data = result[1:].T
print(uniform_data)
ax = sns.heatmap(uniform_data)
# 设置横坐标的刻度位置和标签
plt.xticks(ticks=np.arange(uniform_data.shape[1]), labels=[f'{i/1000}k' for i in x_bins])

# 设置纵坐标的刻度位置和标签
plt.yticks(ticks=np.arange(uniform_data.shape[0]), labels=[f'{j/10}' for j in range(10)])

# 旋转刻度标签以提高可读性
plt.xticks(rotation=90)  # 横坐标标签旋转90度
plt.yticks(rotation=0)   # 纵坐标标签保持不旋转

# 保存热力图到指定文件路径
plt.savefig(file_path)
plt.savefig(file_path)

