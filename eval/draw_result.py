import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from tools import remove_special_chars, has_word, VQAEval

tiankong = VQAEval()

def parse_english_number_to_int(word):
    """Converts common English number words to their integer values."""
    mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    word = word.lower().strip()
    return mapping.get(word, None)

def yes_or_no(answer, response):
    if isinstance(answer, int):
        if response.isdigit():
            return int(response) == answer
        else:
            # Convert English word to number
            numeric_value = parse_english_number_to_int(response)
            if numeric_value is not None:
                return numeric_value == answer
            else:
                # Try parsing compound numbers
                parts = response.lower().split()
                if len(parts) > 1:
                    # Example: 'twenty one' -> 21
                    total = 0
                    for part in parts:
                        value = parse_english_number_to_int(part)
                        if value is None:
                            return False
                        total = total * 10 if total >= 20 else total + value
                    return total == answer
                return False
    else:
        return tiankong.evaluate(response, answer)

# x 划分的区间
x_bins = [1000, 2000, 3000, 5000, 9000, 15000, 26000, 44000, 75000]
# y 的划分档位
y_interval = 0.2

result_path_list = os.listdir('/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer')
# result_path_list = os.listdir('/mnt/petrelfs/renyiming/dataset/sea-needle/eval/ans_tem')
for file_name in result_path_list:
    # 初始化二维数组，x 划分为5档，y 划分为10档（因为 0-1 划分为0.1的10个区间）
    total = np.zeros((len(x_bins) + 1, int(1 / y_interval)))
    correct = np.zeros((len(x_bins) + 1, int(1 / y_interval)))
    # jsonl 文件路径
    jsonl_file_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer/' + file_name
    # jsonl_file_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/ans_tem/' + file_name
    file_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/result/' + file_name[:-4] + 'v2' + '.jpg'

    # 读取 jsonl 文件
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            x = entry['total_tokens']
            y = entry['position']
            if isinstance(y, list):
                y = entry['position'][0]
            else:
                y = entry['position']

            if 'infer-choose' in file_name or 'visual-reasoning' in file_name:
                y = sum(entry['position']) / len(entry['position'])
                
            if y == 1.0:
                y = 0.99

            z = entry['response']
            answer = entry['answer']

            # 确定 x 的档位
            x_index = np.digitize(x, x_bins)
            # 确定 y 的档位
            y_index = int(y / y_interval)
            # y_index = 0
            try:
                # 将 z 值加到对应的档位中
                total[x_index][y_index] += 1
            except Exception as e:
                print(e)
                print(file_name)
                print(y)
                print(x)
                print(answer)
                print(z)
                print('\n')

            if yes_or_no(answer, z):
                if y > 0.5:
                    print(file_name)
                    print(y)
                    print(x)
                    print(answer)
                    print(z)
                    print('\n')
                correct[x_index][y_index] += 1

        result = np.divide(correct, total, out=np.zeros_like(correct), where=total != 0)

    # 打印结果
    print(result)

    # # Plot a heatmap for a numpy array:
    uniform_data = result[1:].T
    print(uniform_data)

    # Define the custom color map
    from matplotlib.colors import LinearSegmentedColormap

    colors = colors = ["#DC143C", "#FFD700", "#3CB371"]  # Red to Yellow to Green
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    ax = sns.heatmap(uniform_data, vmin=0, vmax=1, cmap=cm)

    # 设置横坐标的刻度位置和标签
    plt.xticks(ticks=np.arange(uniform_data.shape[1])+0.5, labels=[f'{i / 1000}k' for i in x_bins])

    # 设置纵坐标的刻度位置和标签
    plt.yticks(ticks=np.arange(uniform_data.shape[0]), labels=[f'{j / (1/y_interval)}' for j in range(int(1/y_interval))])

    # 旋转刻度标签以提高可读性
    plt.xticks(rotation=90)  # 横坐标标签旋转90度
    plt.yticks(rotation=0)   # 纵坐标标签保持不旋转

    # 保存热力图到指定文件路径
    plt.savefig(file_path)
    plt.clf()
