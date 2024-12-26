import os
import math
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from scale_margin_train import yoloxdstre
# from eval_voc_scales_v1 import eval_voc_scale_v1
# from eval_voc_scales_v2 import eval_voc_scale_v2

matplotlib.use('Agg')


def visualize_margin_curve(data_x, data_y, data_label, title, x_label, y_label, x_range, y_range, save_path, show_point=False):
    # setting up font
    font = {'family' : 'Times New Roman', 'size' : 12}
    plt.rc('font', **font)
    
    color16 = "#D2AA3A" # 科研黄
    color45 = "#D94738" # 科研红
    color55 = "#039F89" # 科研绿
    color77 = "#A04294" # 科研紫

    color_list = [color16, color45, color55]

    # setting up background & lines
    sns.set_style("whitegrid") # 设置背景样式
    for i in range(len(data_y)):
        sns.lineplot(x=data_x, y=data_y[i], color=color_list[i], linewidth=1, marker="o", markersize=2, markeredgecolor=color_list[i], markeredgewidth=1, label=data_label[i])
        if i == 0: plt.axvline(x=90, ymax=0.30, color=color16, linestyle='--', linewidth=2, label='$Small ~ Best ~ Generalization$')
        if i == 1: plt.axvline(x=190, ymax=0.44, color=color45, linestyle='--', linewidth=2, label='$Middle ~ Best ~ Generalization$')
        if i == 2: plt.axvline(x=230, ymax=0.55, color=color55, linestyle='--', linewidth=2, label='$Large ~ Best ~ Generalization$')
    
        # annotate exact data on the line
        if show_point:
            for j in range(len(data_x)):
                if i == 2:
                    if j == 1:
                        if j % 2 == 0: coor = (-3, 8.5)
                        if j % 2 == 1: coor = (6, -12.5)
                    else:
                        if j % 2 == 0: coor = (-2, 8.5)
                        if j % 2 == 1: coor = (4, -12.5)
                else:
                    if j % 2 == 0: coor = (-2, 8.5)
                    if j % 2 == 1: coor = (4, -12.5)

                # plt.annotate(f'{data_x[j]}, {data_y[i][j]:.2f}', (data_x[j], data_y[i][j]), fontsize=9, textcoords="offset points", 
                # xytext=coor, ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))

                plt.annotate(f'{data_y[i][j]:.2f}', (data_x[j], data_y[i][j]), fontsize=9, textcoords="offset points", 
                xytext=coor, ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))

                # if i == 0 and j == 17:
                #     plt.annotate(f'{data_x[j]}, {data_y[i][j]:.2f}', (data_y[i][j]), fontsize=9, textcoords="offset points", 
                #     xytext=coor, ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
                # if i == 1 and j == 18:
                #     plt.annotate(f'{data_x[j]}, {data_y[i][j]:.2f}', (data_y[i][j]), fontsize=9, textcoords="offset points", 
                #     xytext=coor, ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
                # if i == 2 and j == 22:
                #     plt.annotate(f'{data_x[j]}, {data_y[i][j]:.2f}', (data_y[i][j]), fontsize=9, textcoords="offset points", 
                #     xytext=coor, ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))

    # 最佳泛化点竖线
    # plt.axvline(x=200, color=color77, linestyle='--', linewidth=1.5, label='$Overall ~ Best ~ Generalization$')

    # setting up title
    plt.title(title, fontweight='bold', fontsize=12)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    
    # setting up label
    plt.legend(loc='upper left', frameon=True, fontsize=9)

    # 设置刻度字体和范围
    data_x = [50, 90, 100, 150, 190, 200, 230, 240]
    plt.xticks(ticks=data_x, fontsize=8)
    plt.yticks(fontsize=8)
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

    print(f"ratio: {rat_list}")
    print('\n')
    print(f"scm list: {scm_list}")
    print('\n')
    print(f"scb list: {scb_list}")
    print('\n')
    print(f"sca list: {sca_list}")
    print('\n')

def visualize_control(data_list):
    
    data_x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 
              140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    
    data_y = data_list

    title = 'Test Set Accuracy'

    x_range = [0, 250] # 
    y_range = [0, 150] # 

    x_label, y_label = 'Epoch', 'Average Precision'

    data_label = ['$Training ~ Small ~ Object ~ AP_{s}$', 
                  '$Training ~ Middle ~ Object ~ AP_{m}$', 
                  '$Training ~ Large ~ Object ~ AP_{l}$']
    
    save_path = '/home/uic/ChengYuxuan/YoloXDSTRe1/tblogger/' + 'test_accuracy' #  # train_accuracy

    visualize_margin_curve(data_x, data_y, data_label, title, x_label, y_label, x_range, y_range, save_path, show_point=True)


if __name__ == '__main__':
    # single_point, from_zero = False, True

    # test one or all weights
    # if single_point:
    #     ckpt_list = ['epoch_' + str(test_epoch) + '_ckpt.pth'] # for test usage
    # else:
    #     ckpt_list = []
    #     for i in range(10, 241, 10):
    #         ckpt = 'epoch_' + str(i) + '_ckpt.pth'
    #         ckpt_list.append(ckpt)

    # weight path
    # baseline = '/home/uic/ChengYuxuan/' + 'YoloXDSTRe3/YOLOX_outputs/' + 'Baseline_S/' #  # 'build_rewrite/' # 'build_rewrite/'

    # train & test adaptive bins
    # main_finetune(baseline, ckpt_list, from_zero)

    small = [20.64, 27.49, 30.63, 34.47, 36.07, 36.63, 38.76, 38.88, 40.74, 40.71, 39.53, 41.23, 40.46, 39.97, 41.02, 41.58, 41.42, 42.18, 41.26, 40.82, 41.46, 41.25, 40.74, 40.67]
    midle = [40.55, 49.26, 53.13, 56.77, 59.05, 60.70, 61.84, 62.42, 63.46, 64.11, 63.92, 65.36, 65.49, 65.90, 66.36, 66.81, 66.77, 67.23, 67.55, 67.52, 66.22, 66.69, 66.78, 66.51]
    large = [59.71, 69.28, 73.54, 76.72, 78.34, 80.01, 80.70, 81.03, 81.66, 82.43, 82.43, 83.22, 83.46, 83.70, 84.11, 84.71, 84.50, 84.83, 84.97, 85.10, 85.23, 85.32, 85.42, 85.42]

    visualize_control([small, midle, large])

    # baseline train margin
    # small = [0.46, 0.62, 1.23, 1.66, 1.86, 2.27, 2.14, 2.37, 2.44, 2.82, 2.63, 3.01, 2.86, 3.29, 3.36, 3.61, 3.69, 3.87, 4.08, 3.95, 3.76, 4.620, 4.700, 5.040]
    # midle = [2.43, 3.34, 4.02, 4.75, 5.34, 5.68, 5.85, 6.14, 6.19, 6.67, 6.54, 6.71, 7.04, 7.32, 7.72, 7.83, 7.95, 8.41, 8.50, 8.77, 8.61, 9.160, 8.960, 9.230]
    # large = [3.01, 4.14, 4.99, 5.51, 6.16, 6.66, 7.02, 7.23, 7.33, 7.67, 8.10, 7.87, 8.40, 8.35, 9.01, 8.94, 9.04, 9.35, 9.56, 9.80, 9.94, 10.05, 10.26, 10.35]

    # baseline test margin
    # small = [0.36, 0.59, 0.73, 0.86, 0.96, 1.02, 1.05, 1.09, 1.13, 1.17, 1.2, 1.24, 1.25, 1.28, 1.29, 1.33, 1.36, 1.4, 1.42, 1.44, 1.46, 1.47, 1.5, 1.5]
    # midle = [1.85, 2.81, 3.31, 3.78, 4.07, 4.28, 4.47, 4.55, 4.67, 4.77, 4.84, 4.94, 5.03, 5.13, 5.15, 5.2, 5.3, 5.39, 5.45, 5.5, 5.52, 5.53, 5.58, 5.6]
    # large = [3.96, 5.52, 6.58, 7.49, 8.02, 8.43, 8.7, 8.8, 9.07, 9.23, 9.27, 9.4, 9.5, 9.56, 9.67, 9.71, 9.75, 9.85, 9.88, 10.0, 10.02, 10.04, 10.03, 10.11]
    # total = [3.67, 5.21, 6.19, 7.06, 7.57, 7.96, 8.24, 8.36, 8.63, 8.78, 8.85, 9.0, 9.11, 9.22, 9.29, 9.37, 9.44, 9.55, 9.6, 9.71, 9.74, 9.76, 9.78, 9.85]

    # baseline test ap
    
    
