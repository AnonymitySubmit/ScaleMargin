import os
import math, json
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from scale_margin_train import yoloxdstre
# from eval_voc_scales_v1 import eval_voc_scale_v1
# from eval_voc_scales_v2 import eval_voc_scale_v2

matplotlib.use('Agg')

color16 = "#D2AA3A" # 科研黄
color45 = "#D94738" # 科研红
color55 = "#039F89" # 科研绿
color61 = "#1C6AB1" # 科研蓝
color77 = "#A04294" # 科研紫
color88 = "#F7C2CD" # 科研粉

def get_parallel_box(point1, point2, height):
    # 确保point1和point2在y=x线上
    assert point1[0] == point1[1], "point1不在y=x线上"
    assert point2[0] == point2[1], "point2不在y=x线上"
    
    # 计算斜率
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    # 因为在y=x线上，斜率应该是1，这里只是验证
    assert slope == 1.0 or (point1[0] == point2[0] and slope == float('inf')), "point1和point2的连线不是y=x方向"
    
    # 计算中点
    mid_x = (point1[0] + point2[0]) / 2
    mid_y = (point1[1] + point2[1]) / 2
    
    # 计算方框的四个顶点
    # 上顶点: (mid_x - height/sqrt(2), mid_y + height/sqrt(2))
    # 下顶点: (mid_x + height/sqrt(2), mid_y - height/sqrt(2))
    # 由于平行于y=x线，左右顶点通过旋转中点坐标得到
    import math
    half_height = height / math.sqrt(2)
    
    top_left = (mid_x - half_height, mid_y + half_height)
    top_right = (mid_x + half_height, mid_y + half_height)
    bottom_left = (mid_x - half_height, mid_y - half_height)
    bottom_right = (mid_x + half_height, mid_y - half_height)
    
    box_points = [top_left, top_right, bottom_left, bottom_right]
    return box_points

def visualize_margin_curve(data_x, data_y, data_label, title, x_label, y_label, x_range, y_range, save_path, n, show_point=False):
    # setting up font
    font = {'family' : 'Times New Roman', 'size' : 16}
    plt.rc('font', **font)

    # setting up background & lines
    sns.set_style("whitegrid") # 设置背景样式

    baseline = r'$\delta_{s}=0, \delta_{m}=0, \delta_{l}=0$'
    optimal = r'$\delta_{s}=1, \delta_{m}=\frac{1}{2}, \delta_{l}=\frac{1}{4}$'

    import pandas as pd
    data_0 = pd.DataFrame({'x': data_x[0],
                           'y': data_y[0],
                           'label':baseline})

    data_1 = pd.DataFrame({'x': data_x[1],
                           'y': data_y[1],
                           'label':optimal})

    combined_data = pd.concat([data_0, data_1])
    custom_palette = {baseline: color61, optimal: color45}

    sns.scatterplot(x='x', y='y', data=combined_data, hue='label', palette=custom_palette, s=18, alpha=0.8)

    # 最佳泛化点竖线
    # plt.axvline(x=240, color=color77, linestyle='--', linewidth=1, label='$Overall ~ Best ~ Generalization$')

    if n == 0:
        color_list = ['grey', 'grey', 'grey', 'grey', color16, 'grey', 'grey', 'grey']
    if n == 1:
        color_list = ['grey', 'grey', 'grey', 'grey', color61, 'grey', 'grey', 'grey']
    if n == 2:
        color_list = ['grey', 'grey', 'grey', color61, color61, 'grey', 'grey', 'grey']
    if n == 3:
        color_list = ['grey', 'grey', color61, color61, color16, 'grey', 'grey', 'grey']
    if n == 4:
        color_list = ['grey', color61, color61, 'grey', color16, 'grey', 'grey', 'grey']
    if n == 5:
        color_list = [color61, color61, 'grey', 'grey', color16, 'grey', 'grey', 'grey']

    # 环带对角线, x range, y range
    plt.plot([-50, 30], [-30, 50], color=color_list[0], linestyle='--', linewidth=2.5) # y = x + 15
    plt.plot([-45, 30], [-30, 45], color=color_list[1], linestyle='--', linewidth=2.5) # y = x + 15
    plt.plot([-40, 30], [-30, 40], color=color_list[2], linestyle='--', linewidth=2.5) # y = x + 10
    plt.plot([-35, 30], [-30, 35], color=color_list[3], linestyle='--', linewidth=2.5) # y = x + 5
    plt.plot([-30, 30], [-30, 30], color=color_list[4], linestyle='--', linewidth=2.5) # y = x 
    # plt.plot([-25, 30], [-30, 25], color=color_list[5], linestyle='--', linewidth=2.5) # y = x - 5
    # plt.plot([-20, 30], [-30, 20], color=color_list[6], linestyle='--', linewidth=2.5) # y = x - 10
    # plt.plot([-15, 30], [-30, 15], color=color_list[7], linestyle='--', linewidth=2.5) # y = x - 15

    # setting up title
    plt.title(title, fontsize=16) # , fontweight='bold'
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    # if n == 1:
    #     start_x, end_x = -3, -1.7 # 箭头的起点
    #     start_y, end_y = -0.4, -0.4 # 箭头的终点

    # if n == 2:
    #     start_x, end_x = -8.6, -7.3 # 箭头的起点
    #     start_y, end_y = 1, 1 # 箭头的终点

    # if n == 3:
    #     start_x, start_y = -11.5, 0 # 箭头的起点
    #     end_x, end_y = -10.8, 0.7 # 箭头的终点

    # if n == 4:
    #     start_x, end_x = -14, -14 # 箭头的起点
    #     start_y, end_y = 4.0, 3.3 # 箭头的终点

    # 圈起optimal超过baseline的点(-0.5, 0.2), (-1.3, -0.3), (0, 0), (-1, -1)
    # if n != 0 and n!= 5:
    #     plt.arrow(start_x, start_y, end_x-start_x, end_y-start_y, linewidth=3, head_width=0.3, head_length=0.3, fc=color45, ec=color45)

    # setting up label
    plt.legend(loc='lower left', frameon=True, fontsize=16)

    # 设置刻度字体和范围
    data_x = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    data_y = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    plt.xticks(fontsize=14) # ticks=data_x
    plt.yticks(fontsize=14) # ticks=data_y
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    
    # 设置坐标轴样式
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)

    # 调整布局
    plt.tight_layout()

    plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight')

    # plt.show()

def visualize_margin_histogram(data_x, data_y, data_label, title, x_label, y_label, x_range, y_range, save_path):
    fig, axs = plt.subplots(1, 1, figsize=(16, 8), sharex=True) # figure  

    # setting up font
    font = {'family' : 'Times New Roman', 'size' : 12}
    plt.rc('font', **font)

    # 设置边框和网格
    # for ax in axs:
    #     ax.spines['top'].set_color('gray')
    #     ax.spines['bottom'].set_color('gray')
    #     ax.spines['left'].set_color('gray')
    #     ax.spines['right'].set_color('gray')
    #     ax.spines['top'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_linewidth(1.5)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['right'].set_linewidth(1.5)
    #     ax.grid(color='gray', linestyle='--', linewidth=0.5) # 设置灰色网格
    
    axs.spines['top'].set_color('gray')
    axs.spines['bottom'].set_color('gray')
    axs.spines['left'].set_color('gray')
    axs.spines['right'].set_color('gray')
    axs.spines['top'].set_linewidth(1.5)
    axs.spines['bottom'].set_linewidth(1.5)
    axs.spines['left'].set_linewidth(1.5)
    axs.spines['right'].set_linewidth(1.5)
    axs.grid(color='gray', linestyle='--', linewidth=0.5) # 设置灰色网格
    
    # 绘制第一个子图
    # axs[0].plot(x_values, data_y[0], 'b-', linewidth=0.4, marker="o", markersize=0.5, markeredgecolor='white', markeredgewidth=0.5)
    # axs[0].set_ylabel('$Scale ~ Margin ~ C_s$', fontsize=10, labelpad=2.5)
    # axs[0].set_ylim(0.0, 0.5)
    # axs[0].set_xlim(0, 250)
    # axs[0].grid(True, which='both', color='grey', linestyle='-')

    # 绘制第二个子图
    # axs[1].plot(x_values, data_y[1], 'r-', linewidth=0.4, marker="o", markersize=0.5, markeredgecolor='white', markeredgewidth=0.5)
    # axs[1].set_ylabel('$Scale ~ Balance ~ C_s$', fontsize=10, labelpad=12)
    # axs[1].set_ylim(0.0, 2.0)
    # axs[1].set_xlim(0, 250)
    # axs[1].grid(True, which='both', color='grey', linestyle='-', )

    # 绘制第三个子图
    # axs[2].plot(x_values, data_y[2], 'g-', linewidth=0.4, marker="o", markersize=0.5, markeredgecolor='white', markeredgewidth=0.5)
    # axs[2].set_ylabel('$Scale ~ Adaptive ~ C_s$', fontsize=10, labelpad=5)
    # axs[2].set_ylim(-1, 30)
    # axs[2].set_xlim(0, 250)
    # axs[2].grid(True, which='both', color='grey', linestyle='-')

    # 设定并创建间隔列表
    step = math.sqrt(5)/5
    n = int((x_range[1] - x_range[0]) / step)
    ranges = [(x_range[0] + i*step, x_range[0] + (i+1)*step) for i in range(n)]

    print(f"n: {n}, ranges: {len(ranges)}")

    # 处理最后一个范围可能超出end的问题（浮点误差）
    # ranges[-1] = (ranges[-1][0], x_range[1])

    # 初始化数据量列表
    data_x_num = [0 for _ in range(len(ranges))]
    data_y_num = [0 for _ in range(len(ranges))]

    # 计算每个间隔内的数据量
    for i in data_x: # 每个数据点
        for j in range(len(ranges)): # 每个范围
            if ranges[j][1] > i >= ranges[j][0]:
                data_x_num[j] += 1 # 数据处于范围则+1
    
    for i in data_y:
        for j in range(len(ranges)):
            if ranges[j][1] > i >= ranges[j][0]:
                data_y_num[j] += 1

    # 计算间隔在x轴上的坐标
    x_coor = [ranges[i][0] + (ranges[i][1]-ranges[i][0])/2 for i in range(len(ranges))]

    # 创建柱状图
    bar1 = axs.bar(x_coor, data_x_num, width=0.4, label='Baseline', color=color61)
    bar2 = axs.bar(x_coor, data_y_num, width=0.4, label='Optimal', color=color45, alpha=0.5)

    axs.set_xlim(x_range[0], x_range[1])
    axs.set_ylim(y_range[0], y_range[1])

    # 调整布局
    plt.tight_layout()

    plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight')

    # plt.show()

def retain_decimal(list_, num):
    for i in range(len(list_)):
        list_[i] = round(list_[i], num)
    return list_

def main_finetune(baseline, ckpt_list, from_zero):
    rat_list, scm_list, scb_list, sca_list = [], [], [], []

    if from_zero: # ckpt_path, resume
        data_list = yoloxdstre(baseline + ckpt_list[0], False)
        rat_list.append(retain_decimal(data_list[0], 2))
        scm_list.append(retain_decimal(data_list[1], 2))
        scb_list.append(retain_decimal(data_list[2], 2))
        sca_list.append(retain_decimal(data_list[3], 2))

    # fine-tuning for certain rounds
    for i in range(len(ckpt_list)):
        data_list = yoloxdstre(baseline + ckpt_list[i], True)
        rat_list.append(retain_decimal(data_list[0], 2))
        scm_list.append(retain_decimal(data_list[1], 2))
        scb_list.append(retain_decimal(data_list[2], 2))
        sca_list.append(retain_decimal(data_list[3], 2))

    print(f"ratio = {rat_list}\n")
    print(f"scm list = {scm_list}\n")
    print(f"scb list = {scb_list}\n")
    print(f"sca list = {sca_list}\n")

def visualize_control(data_x, data_y, pair, range_list):
    title = 'Test Set Small Object Logit Distribution'

    n = 1

    if n == 0:
        title = 'Test Set Small Object Logit Distribution'
        x_range, y_range = [-17.5, 2.5], [-9, 4]
    elif n == 1:
        title = r'$Baseline ~ Object ~ Predictions ~ in ~ Belt ~ Area ~ [-∞, 0]$'
        x_range, y_range = [-13, 5], [-9, 1]
    elif n == 2:
        title = r'$Baseline ~ Object ~ Predictions ~ in ~ Belt ~ Area ~ [0, 5]$'
        x_range, y_range = [-13.5, 4.5], [-7, 3]
    elif n == 3:
        title = r'$Baseline ~ Object ~ Predictions ~ in ~ Belt ~ Area ~ [5, 10]$'
        x_range, y_range = [-14, 4], [-6, 4]
    elif n == 4:
        title = r'$Baseline ~ Object ~ Predictions ~ in ~ Belt ~ Area ~ [10, 15]$'
        x_range, y_range = [-16.5, 1.5], [-4, 6]
    elif n == 5:
        title = r'$Baseline ~ Object ~ Predictions ~ in ~ Belt ~ Area ~ [15, 20]$'
        x_range, y_range = [-17.2, 1], [-4, 6]
    else:
        x_range = [-20, 5] # [-20, 5] # [-30, 30] # [-10*math.sqrt(5), 20*math.sqrt(5)]
        y_range = [-10, 5] # [-15, 5] # [-30, 30] # [0, 40]

    if pair:
        x_label, y_label = 'Max Wrong Logit', 'Correct Logit'
    else:
        x_label, y_label = 'Margin', 'Number'

    data_label = ['$Baseline ~ Small ~ Object ~ AP_{s}$', 
                  '$\delta_{s}=1 ~ Small ~ Object ~ AP_{s}$', 
                  '$\delta_{s}=1, \delta_{m}=0.5 ~ Small ~ Object ~ AP_{s}$', 
                  '$\delta_{s}=1, \delta_{m}=0.5, \delta_{m}=0.25 ~ Small ~ Object ~ AP_{s}$',]

    if pair:
        save_path = '/home/uic/ChengYuxuan/YoloXDSTRe1/tblogger/' + f'test_distribution_{n}'
    else:
        save_path = '/home/uic/ChengYuxuan/YoloXDSTRe1/tblogger/' + f'test_distribution_{n}' # test_accuracy # train_margin

    if pair:
        x, y = [data_x[0][n], data_x[1][n]], [data_y[0][n], data_y[1][n]]
        visualize_margin_curve(x, y, data_label, title, x_label, y_label, x_range, y_range, save_path, n, show_point=False)
    else:
        visualize_margin_histogram(data_x[n], data_y[n], data_label, title, x_label, y_label, x_range, y_range, save_path)

def read_in_data(prefix, double_lists):
    # read in data from txt
    name_list = ['small', 'midle', 'large']
    correct_logits = [[] for _ in name_list]
    max_wrong_logits = [[] for _ in name_list]

    for i in range(len(name_list)):
        if double_lists == False:
            # 从txt读取单层列表
            name = prefix + name_list[i] + '_correct_logits.txt'
            with open(name, 'r') as file:
                for line in file:
                    correct_logits[i].append(float(line.strip()))
            name = prefix + name_list[i] + '_max_wrong_logits.txt'
            with open(name, 'r') as file:
                for line in file:
                    max_wrong_logits[i].append(float(line.strip()))
        else:
            # 从txt读取双层列表
            name = prefix + name_list[i] + '_correct_logits.txt'
            with open(name, 'r') as file:
                for line in file:
                    correct_logits[i].append(list(map(float, line.strip().split())))
                # correct_logits[i] = json.load(file)
            name = prefix + name_list[i] + '_max_wrong_logits.txt'
            with open(name, 'r') as file:
                for line in file:
                    max_wrong_logits[i].append(list(map(float, line.strip().split())))
                # max_wrong_logits[i] = json.load(file)
    
    return correct_logits, max_wrong_logits

def compute_belt_num(correct, max_wrong):
    belt_0, belt_0_5, belt_5_10, belt_10_15 = 0, 0, 0, 0

    for i in range(len(max_wrong)):
        if 0 < (correct[i] - max_wrong[i]):
            belt_0 += 1
        if -5 < (correct[i] - max_wrong[i]) <= 0:
            belt_0_5 += 1
        if -10 < (correct[i] - max_wrong[i]) <= -5:
            belt_5_10 += 1
        if -15 < (correct[i] - max_wrong[i]) <= -10:
            belt_10_15 += 1
    
    return belt_0, belt_0_5, belt_5_10, belt_10_15

def print_belt_data(name, obj_length, belt_0, belt_0_5, belt_5_10, belt_10_15):
    print(f"{name}")
    print(f"belt  above  0: {round(belt_0 * 100 / obj_length, 2)}")
    print(f"belt 0   ~ - 5: {round(belt_0_5 * 100 / obj_length, 2)}")
    print(f"belt -5  ~ -10: {round(belt_5_10 * 100 / obj_length, 2)}")
    print(f"belt -10 ~ -15: {round(belt_10_15 * 100 / obj_length, 2)}\n")

def compute_avg_margin(ckpt_list, main_path, file_name, double_lists):
    small_avg_correct, midle_avg_correct, large_avg_correct = [], [], []
    small_avg_wrong, midle_avg_wrong, large_avg_wrong = [], [], []

    for i in ckpt_list:
        margin_correct, margin_wrong = [[], [], []], [[], [], []]
        bl_correct_logits, bl_max_wrong_logits = read_in_data(main_path + file_name + i + '_', double_lists)
        for j in range(len(bl_correct_logits)):
            for m in range(len(bl_correct_logits[j])):
                if bl_correct_logits[j][m] >= bl_max_wrong_logits[j][m]:
                    margin_correct[j].append(bl_correct_logits[j][m]-bl_max_wrong_logits[j][m])
                else:
                    margin_wrong[j].append(bl_correct_logits[j][m]-bl_max_wrong_logits[j][m])

        small_avg_correct.append(round(sum(margin_correct[0]) / len(margin_correct[0]), 2))
        midle_avg_correct.append(round(sum(margin_correct[1]) / len(margin_correct[1]), 2))
        large_avg_correct.append(round(sum(margin_correct[2]) / len(margin_correct[2]), 2))

        small_avg_wrong.append(round(sum(margin_wrong[0]) / len(margin_wrong[0]), 2))
        midle_avg_wrong.append(round(sum(margin_wrong[1]) / len(margin_wrong[1]), 2))
        large_avg_wrong.append(round(sum(margin_wrong[2]) / len(margin_wrong[2]), 2))

    print(f"Small Correct Avg Margin = {small_avg_correct}")
    print(f"Small MaxWrong Avg Margin = {small_avg_wrong}\n")

    print(f"Midle Correct Avg Margin = {midle_avg_correct}")
    print(f"Midle MaxWrong Avg Margin = {midle_avg_wrong}\n")

    print(f"Large Correct Avg Margin = {large_avg_correct}")
    print(f"Large MaxWrong Avg Margin = {large_avg_wrong}\n")

def extract_pari_logits(bl_corrt, bl_max_wrg, op_corrt, op_max_wrg, range_1, range_2, pair):

    def maximum(list_1, list_2):
        margin_list = []
        for i in range(len(list_1)):
            margin_list.append(list_1[i] - list_2[i])
        return max(margin_list)
    
    def filter_max(correct, max_wrng, range_1, range_2):
        if isinstance(range_1, str):
            if maximum(correct, max_wrng) < 0:
                return True
            else:
                return False
        else:
            if range_2 > maximum(correct, max_wrng) > range_1:
                return True
            else:
                return False
    
    def filter_base(correct, max_wrng, range_1, range_2):
        if isinstance(range_1, str):
            if correct - max_wrng < 0:
                return True
            else:
                return False
        else:
            if range_2 > (correct - max_wrng) > range_1:
                return True
            else:
                return False
    
    def filter_opti(correct, max_wrng, range_1, range_2):
        if isinstance(range_1, str):
            if correct - max_wrng > -5:
                return True
            else:
                return False
        else:
            if correct - max_wrng > range_1:
                return True
            else:
                return False

    bl_correct, bl_max_wrong = [], []
    op_correct, op_max_wrong = [], []
    bl_histogr, op_histogr = [], []
    bl_lost_obj, op_lost_obj = 0, 0
    
    # Baseline & Optimal: Baseline中最大边距小于0的目标
    for i in range(len(bl_corrt)):
        if bl_corrt[i] == "occupy" and op_corrt[i] != "occupy": 
            bl_lost_obj += 1
            continue
        
        if bl_corrt[i] != "occupy":
            if filter_max(bl_corrt[i], bl_max_wrg[i], range_1, range_2):
                if op_corrt[i] == "occupy":
                    op_lost_obj += 1
                else:
                    if pair: # scatter point figure
                        # unfilter less zero prediction
                        # bl_correct.append(bl_corrt[i])
                        # bl_max_wrong.append(bl_max_wrg[i])
                        # op_correct.append(op_corrt[i])
                        # op_max_wrong.append(op_max_wrg[i])

                        # filter out less zero prediction
                        for j in range(len(bl_corrt[i])):
                            if filter_base(bl_corrt[i][j], bl_max_wrg[i][j], range_1, range_2):
                                bl_correct.append(bl_corrt[i][j])
                                bl_max_wrong.append(bl_max_wrg[i][j])
                        for j in range(len(op_corrt[i])):
                            if filter_opti(op_corrt[i][j], op_max_wrg[i][j], range_1, range_2):
                                op_correct.append(op_corrt[i][j])
                                op_max_wrong.append(op_max_wrg[i][j])
                    else: # histogram figure
                        bl_histogr.extend([bl_corrt[i][j]-bl_max_wrg[i][j] for j in range(len(bl_corrt[i]))])
                        op_histogr.extend([op_corrt[i][j]-op_max_wrg[i][j] for j in range(len(op_corrt[i]))])

    # print(f"Max Margin Range: [{range_1}, {range_2}]")
    # print(f"baseline lost object num = {bl_lost_obj}")
    # print(f"optimal lost object num = {op_lost_obj}")

    if pair:
        return bl_correct, bl_max_wrong, op_correct, op_max_wrong
    else:
        return bl_histogr, op_histogr


if __name__ == '__main__':
    single_point = False
    epoch_number = 190
    from_scratch = True
    double_lists = True

    # test one or all weights
    if single_point:
        ckpt_list = ['epoch_' + str(ckpt_number) + '_ckpt.pth'] # for test usage
    else:
        ckpt_list = []
        for i in range(10, 241, 10):
            ckpt = 'epoch_' + str(i) + '_ckpt.pth'
            ckpt_list.append(ckpt)

    # weight path
    # baseline = '/home/uic/ChengYuxuan/' + 'YoloXDSTRe3/YOLOX_outputs/' + 'Baseline_S/' #  # 'build_rewrite/' # 'build_rewrite/'

    # train & test adaptive bins
    # main_finetune(baseline, ckpt_list, from_scratch)

    # avg correct & wrong margin
    # compute_avg_margin(ckpt_list, "./logits/", 'optimal_train_multi_ckpt/', double_lists)
    # temp = 1
    # assert temp == 2

    main_path = '/home/uic/ChengYuxuan/YoloXDSTRe2/' # ./logits/' # '/home/uic/ChengYuxuan/YoloXDSTRe2/'
    file_bl = 'baseline_test_single_ckpt/' # baseline_train_single_ckpt/ # baseline_test_single_ckpt/
    file_op = 'optimal_test_single_ckpt/' # optimal_train_single_ckpt/ # optimal_test_single_ckpt/
    ckpt_bl = 'epoch_200_ckpt.pth_' # 'baseline_' # 'epoch_200_ckpt.pth_'
    ckpt_op = 'epoch_190_ckpt.pth_' # 'optimal_' # 'epoch_190_ckpt.pth_'

    bl_corrt, bl_max_wrg = read_in_data(main_path + file_bl + ckpt_bl, double_lists)
    op_corrt, op_max_wrg = read_in_data(main_path + file_op + ckpt_op, double_lists)

    # print(len(bl_correct_logits), len(bl_correct_logits[0]), bl_correct_logits[0][0])
    # 1st layer: small, midle, large, 2nd layer: obj, 3th layer: pred

    n, pair = 0, True # 0: small, 1: midle, 2: large | True: scatter, False: histogram

    # belt_0, belt_0_5, belt_5_10, belt_10_15 = compute_belt_num(bl_correct_logits[n], bl_max_wrong_logits[n])
    # print_belt_data("baseline", len(bl_correct_logits[n]), belt_0, belt_0_5, belt_5_10, belt_10_15)

    # belt_0, belt_0_5, belt_5_10, belt_10_15 = compute_belt_num(op_correct_logits[n], op_max_wrong_logits[n])
    # print_belt_data("optimal", len(op_correct_logits[n]), belt_0, belt_0_5, belt_5_10, belt_10_15)

    # print("baseline", len(bl_corrt[n]), len(bl_max_wrg[n]))
    # print("optimal", len(op_corrt[n]), len(op_max_wrg[n]))

    range_list = [['-inf', 0], [0, 5], [5, 10], [10, 15], [15, 20]]

    bl_correct_list, bl_maxwrng_list = [], []
    op_correct_list, op_maxwrng_list = [], []
    bl_histogr_list, op_histogr_list = [], []

    # visualize all objects' pred logits
    if pair:
        temp = []
        for i in range(len(bl_corrt[n])):
            for j in range(len(bl_corrt[n][i])):
                temp.append(bl_corrt[n][i][j])
        bl_correct_list.append(temp)

        temp = []
        for i in range(len(bl_corrt[n])):
            for j in range(len(bl_max_wrg[n][i])):
                temp.append(bl_max_wrg[n][i][j])
        bl_maxwrng_list.append(temp)

        temp = []
        for i in range(len(op_corrt[n])):
            for j in range(len(op_corrt[n][i])):
                temp.append(op_corrt[n][i][j])
        op_correct_list.append(temp)

        temp = []
        for i in range(len(op_corrt[n])):
            for j in range(len(op_max_wrg[n][i])):
                temp.append(op_max_wrg[n][i][j])
        op_maxwrng_list.append(temp)
    else:
        temp = []
        for i in range(len(bl_corrt[n])):
            for j in range(len(bl_corrt[n][i])):
                temp.append(bl_corrt[n][i][j]-bl_max_wrg[n][i][j])
        bl_histogr_list.append(temp)
        print(len(temp))

        temp = []
        for i in range(len(op_corrt[n])):
            for j in range(len(op_corrt[n][i])):
                temp.append(op_corrt[n][i][j]-op_max_wrg[n][i][j])
        print(len(temp))
        op_histogr_list.append(temp)

    for range_i in range_list:
        if pair:
            i_1, i_2, i_3, i_4 = extract_pari_logits(bl_corrt[n], bl_max_wrg[n], op_corrt[n], op_max_wrg[n], range_i[0], range_i[1], pair)
            bl_correct_list.append(i_1)
            bl_maxwrng_list.append(i_2)
            op_correct_list.append(i_3)
            op_maxwrng_list.append(i_4)
        else:
            i_1, i_2 = extract_pari_logits(bl_corrt[n], bl_max_wrg[n], op_corrt[n], op_max_wrg[n], range_i[0], range_i[1], pair)
            bl_histogr_list.append(i_1)
            op_histogr_list.append(i_2)
    
    range_list.insert(0, "all")
    
    if pair:
        visualize_control([bl_maxwrng_list, op_maxwrng_list], [bl_correct_list, op_correct_list], pair, range_list)
    else:
        visualize_control(bl_histogr_list, op_histogr_list, pair, range_list)