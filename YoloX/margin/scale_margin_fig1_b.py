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
    color61 = "#1C6AB1" # 科研蓝
    color77 = "#A04294" # 科研紫

    color_list = [color16, color16, color45, color45, color55, color55, color77]

    # setting up background & lines
    sns.set_style("whitegrid") # 设置背景样式
    for i in range(len(data_y)):
        if i == 0 or i == 2 or i == 4:
            sns.lineplot(x=data_x, y=data_y[i], color=color_list[i], linewidth=1, marker="o", markersize=1, markeredgecolor=color_list[i], markeredgewidth=2, label=data_label[i])
        if i == 1 or i == 3 or i == 5:
            sns.lineplot(x=data_x, y=data_y[i], color=color_list[i], linewidth=1, linestyle='--', marker="D", markersize=1, markeredgecolor=color_list[i], markeredgewidth=2, label=data_label[i])
        # if i == 1: plt.axvline(x=90, ymax=0.16, color=color16, linestyle='--', linewidth=1, label='$Small ~ Best ~ Generalization$')
        # if i == 3: plt.axvline(x=190, ymax=0.55, color=color45, linestyle='--', linewidth=1, label='$Middle ~ Best ~ Generalization$')
        # if i == 5: plt.axvline(x=230, ymax=0.66, color=color55, linestyle='--', linewidth=1, label='$Large ~ Best ~ Generalization$')
    
        # annotate exact data on the line
        if show_point:
            for j in range(len(data_x)):
                if i == 0:
                    if j % 2 == 0: coor = (-2, 10.5)
                    if j % 2 == 1: coor = (2, -12.5)
                if i == 1:
                    if j <= 4:
                        if j % 2 == 0: coor = (0, -12.5)
                        if j % 2 == 1: coor = (2, -12.5)
                    else:
                        if j % 2 == 0: coor = (0, -12.5)
                        if j % 2 == 1: coor = (4, -24.5)
                if i == 2:
                    if j <= 4:
                        if j % 2 == 0: coor = (-1, 10.5)
                        if j % 2 == 1: coor = (-1, 10.5)
                    else:
                        if j % 2 == 0: coor = (1, 10.5)
                        if j % 2 == 1: coor = (1, -12.5)

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
    # plt.axvline(x=200, color=color77, linestyle='--', linewidth=1, label='$Overall ~ Best ~ Generalization$')

    # setting up title
    plt.title(title, fontweight='bold', fontsize=12)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    
    # setting up label
    plt.legend(loc='upper left', frameon=True, fontsize=9)

    # 设置刻度字体和范围
    data_x = [50, 100, 150, 200, 240]
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

    print(f"ratio = {rat_list}\n")
    print(f"scm list = {scm_list}\n")
    print(f"scb list = {scb_list}\n")
    print(f"sca list = {sca_list}\n")

def visualize_control(data_list):
    
    data_x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 
              140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    
    data_y = data_list

    title = 'Test Set Correct & Wrong Margin'

    x_range = [0, 250] # 
    y_range = [-5, 15] # 

    x_label, y_label = 'Epoch', 'Margin'

    # data_label = ['$Training ~ Small ~ Object ~ \overline{\gamma_{s}}$', 
    #               '$Testing ~ Small ~ Object ~ \overline{\gamma_{s}}$', 
    #               '$Training ~ Middle ~ Object ~ \overline{\gamma_{m}}$', 
    #               '$Testing ~ Middle ~ Object ~ \overline{\gamma_{m}}$', 
    #               '$Training ~ Large ~ Object ~ \overline{\gamma_{l}}$',
    #               '$Testing ~ Large ~ Object ~ \overline{\gamma_{l}}$']
    
    data_label = ['$Optimal ~ Correct ~ Small ~ Object ~ \overline{\gamma_{s}}$', 
                  '$Optimal ~ Max ~ Wrong ~ Small ~ Object ~ \overline{\gamma_{s}}$', 
                  '$Baseline ~ Correct ~ Middle ~ Object ~ \overline{\gamma_{m}}$', 
                  '$Baseline ~ Max ~ Wrong ~ Middle ~ Object ~ \overline{\gamma_{m}}$', 
                  '$Correct ~ Large ~ Object ~ \overline{\gamma_{l}}$',
                  '$Max ~ Wrong ~ Large ~ Object ~ \overline{\gamma_{l}}$']
    
    save_path = '/home/uic/ChengYuxuan/YoloXDSTRe1/tblogger/' + 'corrtwrong_train_small' # test_accuracy # train_margin # 

    visualize_margin_curve(data_x, data_y, data_label, title, x_label, y_label, x_range, y_range, save_path, show_point=False)


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

    # optimal train
    op_train_s_corrt = [2.36, 3.08, 3.42, 3.68, 3.87, 4.08, 4.23, 4.36, 4.51, 4.60, 4.74, 4.77, 4.87, 4.93, 5.08, 5.13, 5.29, 5.430, 5.510, 5.64, 5.750, 5.840, 5.970, 6.010]
    op_train_s_maxwr = [-2.33, -2.32, -2.32, -2.34, -2.38, -2.32, -2.33, -2.35, -2.34, -2.36, -2.34, -2.38, -2.34, -2.36, -2.39, -2.35, -2.37, -2.39, -2.34, -2.38, -2.39, -2.38, -2.38, -2.34]

    # optimal test
    op_test_s_corrt = [3.27, 4.06, 4.63, 5.09, 5.39, 5.63, 5.85, 6.01, 6.18, 6.32, 6.44, 6.510, 6.680, 6.780, 6.890, 6.940, 7.130, 7.310, 7.430, 7.590, 7.680, 7.790, 7.940, 7.900]
    op_test_s_maxwr = [-1.73, -1.78, -1.92, -1.98, -2.03, -2.10, -2.19, -2.28, -2.29, -2.38, -2.44, -2.58, -2.48, -2.63, -2.81, -2.75, -2.85, -2.88, -2.90, -2.97, -3.05, -3.06, -3.02, -3.06]

    # baseline train
    bl_train_s_corrt = [2.33, 2.89, 3.35, 3.70, 3.90, 4.08, 4.23, 4.35, 4.44, 4.57, 4.74, 4.76, 4.84, 4.90, 4.95, 5.15, 5.18, 5.35, 5.57, 5.57, 5.68, 5.83, 5.99, 5.94]
    bl_train_s_maxwr = [-2.30, -2.29, -2.33, -2.37, -2.33, -2.35, -2.36, -2.34, -2.35, -2.35, -2.39, -2.36, -2.36, -2.35, -2.35, -2.38, -2.37, -2.39, -2.40, -2.37, -2.36, -2.33, -2.34, -2.34]

    # baseline test
    bl_test_s_corrt = [3.17, 4.00, 4.55, 5.11, 5.46, 5.67, 5.72, 5.90, 6.16, 6.21, 6.36, 6.500, 6.580, 6.710, 6.790, 7.000, 7.130, 7.260, 7.430, 7.560, 7.670, 7.750, 7.950, 7.970]
    bl_test_s_maxwr = [-1.77, -1.81, -1.91, -2.04, -2.12, -2.19, -2.28, -2.27, -2.20, -2.32, -2.34, -2.39, -2.46, -2.54, -2.54, -2.71, -2.74, -2.84, -2.85, -2.93, -2.99, -3.08, -2.96, -3.06]

    visualize_control([op_train_s_corrt, op_train_s_maxwr, bl_train_s_corrt, bl_train_s_maxwr])
    # train_small, test_small, train_midle, test_midle, train_large, test_large
    # test_s_corrt, test_s_maxwr, test_m_corrt, test_m_maxwr, test_l_corrt, test_l_maxwr

    # baseline train margin new codes
    # train_small = [-0.28, 0.45, 0.90, 1.26, 1.54, 1.71, 1.95, 2.10, 2.20, 2.34, 2.49, 2.58, 2.66, 2.87, 3.03, 3.25, 3.24, 3.50, 3.70, 3.79, 3.890, 4.040, 4.150, 4.250]
    # train_midle = [1.79,  3.22, 3.93, 4.66, 5.09, 5.41, 5.71, 5.95, 6.01, 6.26, 6.41, 6.67, 6.82, 7.01, 7.13, 7.39, 7.49, 7.83, 7.91, 8.04, 8.140, 8.340, 8.460, 8.500]
    # train_large = [2.98,  4.42, 5.36, 6.14, 6.66, 7.04, 7.44, 7.63, 7.93, 8.06, 8.22, 8.48, 8.65, 8.93, 8.99, 9.20, 9.33, 9.56, 9.67, 9.86, 10.02, 10.16, 10.16, 10.36]

    # baseline train margin old codes
    # train_small = [0.46, 0.62, 1.23, 1.66, 1.86, 2.27, 2.14, 2.37, 2.44, 2.82, 2.63, 3.01, 2.86, 3.29, 3.36, 3.61, 3.69, 3.87, 4.08, 3.95, 3.76, 4.620, 4.700, 5.040]
    # train_midle = [2.43, 3.34, 4.02, 4.75, 5.34, 5.68, 5.85, 6.14, 6.19, 6.67, 6.54, 6.71, 7.04, 7.32, 7.72, 7.83, 7.95, 8.41, 8.50, 8.77, 8.61, 9.160, 8.960, 9.230]
    # train_large = [3.01, 4.14, 4.99, 5.51, 6.16, 6.66, 7.02, 7.23, 7.33, 7.67, 8.10, 7.87, 8.40, 8.35, 9.01, 8.94, 9.04, 9.35, 9.56, 9.80, 9.94, 10.05, 10.26, 10.35]

    # baseline test margin new codes
    # test_small = [1.63, 2.69, 3.29, 3.89, 4.33, 4.59, 4.72, 4.92, 5.11, 5.28, 5.43, 5.61, 5.65, 5.79, 5.83, 6.01, 6.15, 6.32, 6.40, 6.480, 6.590, 6.650, 6.790, 6.800]
    # test_midle = [3.07, 4.64, 5.47, 6.25, 6.73, 7.08, 7.39, 7.53, 7.73, 7.88, 7.99, 8.16, 8.33, 8.49, 8.51, 8.61, 8.77, 8.91, 9.01, 9.090, 9.130, 9.140, 9.220, 9.260]
    # test_large = [4.00, 5.58, 6.64, 7.57, 8.10, 8.51, 8.79, 8.88, 9.15, 9.32, 9.36, 9.49, 9.59, 9.65, 9.76, 9.81, 9.84, 9.95, 9.98, 10.09, 10.12, 10.13, 10.13, 10.21]

    # baseline test margin old codes
    # test_small = [0.36, 0.59, 0.73, 0.86, 0.96, 1.02, 1.05, 1.09, 1.13, 1.17, 1.2, 1.24, 1.25, 1.28, 1.29, 1.33, 1.36, 1.4, 1.42, 1.44, 1.46, 1.47, 1.5, 1.5]
    # test_midle = [1.85, 2.81, 3.31, 3.78, 4.07, 4.28, 4.47, 4.55, 4.67, 4.77, 4.84, 4.94, 5.03, 5.13, 5.15, 5.2, 5.3, 5.39, 5.45, 5.5, 5.52, 5.53, 5.58, 5.6]
    # test_large = [3.96, 5.52, 6.58, 7.49, 8.02, 8.43, 8.7, 8.8, 9.07, 9.23, 9.27, 9.4, 9.5, 9.56, 9.67, 9.71, 9.75, 9.85, 9.88, 10.0, 10.02, 10.04, 10.03, 10.11]

    # baseline test APs APm APl
    # small = [20.64, 27.49, 30.63, 34.47, 36.07, 36.63, 38.76, 38.88, 40.74, 40.71, 39.53, 41.23, 40.46, 39.97, 41.02, 41.58, 41.42, 42.18, 41.26, 40.82, 41.46, 41.25, 40.74, 40.67]
    # midle = [40.55, 49.26, 53.13, 56.77, 59.05, 60.70, 61.84, 62.42, 63.46, 64.11, 63.92, 65.36, 65.49, 65.90, 66.36, 66.81, 66.77, 67.23, 67.55, 67.52, 66.22, 66.69, 66.78, 66.51]
    # large = [59.71, 69.28, 73.54, 76.72, 78.34, 80.01, 80.70, 81.03, 81.66, 82.43, 82.43, 83.22, 83.46, 83.70, 84.11, 84.71, 84.50, 84.83, 84.97, 85.10, 85.23, 85.32, 85.42, 85.42]
    
    # baseline train correct & wrong avg margin
    # train_s_corrt = [2.33, 2.89, 3.35, 3.70, 3.90, 4.08, 4.23, 4.35, 4.44, 4.57, 4.74, 4.76, 4.84, 4.90, 4.95, 5.15, 5.18, 5.35, 5.57, 5.57, 5.68, 5.83, 5.99, 5.94]
    # train_s_maxwr = [-2.30, -2.29, -2.33, -2.37, -2.33, -2.35, -2.36, -2.34, -2.35, -2.35, -2.39, -2.36, -2.36, -2.35, -2.35, -2.38, -2.37, -2.39, -2.40, -2.37, -2.36, -2.33, -2.34, -2.34]
    # train_m_corrt = [4.06, 5.27, 5.95, 6.41, 6.78, 7.03, 7.31, 7.51, 7.63, 7.73, 7.86, 8.04, 8.17, 8.33, 8.39, 8.55, 8.75, 8.90, 9.06, 9.19, 9.28, 9.38, 9.46, 9.5]
    # train_m_maxwr = [-2.04, -2.07, -2.10, -2.14, -2.09, -2.12, -2.15, -2.13, -2.12, -2.17, -2.19, -2.17, -2.17, -2.16, -2.18, -2.22, -2.18, -2.16, -2.18, -2.17, -2.18, -2.17, -2.18, -2.16]
    # train_l_corrt = [4.69, 5.93, 6.75, 7.46, 7.79, 8.22, 8.44, 8.61, 8.89, 9.01, 9.08, 9.28, 9.44, 9.61, 9.75, 9.91, 9.99, 10.13, 10.26, 10.41, 10.55, 10.67, 10.72, 10.78]
    # train_l_maxwr = [-1.77, -1.81, -1.86, -1.87, -1.88, -1.88, -1.93, -1.90, -1.94, -1.96, -1.92, -1.94, -1.96, -1.95, -1.94, -1.96, -1.92, -2.00, -2.00, -1.94, -1.95, -2.02, -1.96, -2.00]

    # baseline test correct & wrong avg margin
    # test_s_corrt = [3.17, 4.00, 4.55, 5.11, 5.46, 5.67, 5.72, 5.90, 6.16, 6.21, 6.36, 6.500, 6.580, 6.710, 6.790, 7.000, 7.130, 7.260, 7.430, 7.560, 7.670, 7.750, 7.950, 7.970]
    # test_m_corrt = [4.58, 6.01, 6.67, 7.39, 7.82, 8.10, 8.40, 8.52, 8.73, 8.85, 8.98, 9.130, 9.250, 9.430, 9.450, 9.600, 9.720, 9.860, 9.950, 10.03, 10.09, 10.14, 10.23, 10.29]
    # test_l_corrt = [5.49, 6.81, 7.70, 8.51, 8.91, 9.23, 9.48, 9.55, 9.78, 9.90, 9.92, 10.05, 10.12, 10.19, 10.25, 10.31, 10.35, 10.47, 10.48, 10.61, 10.64, 10.67, 10.67, 10.76]
    # test_s_maxwr = [-1.77, -1.81, -1.91, -2.04, -2.12, -2.19, -2.28, -2.27, -2.20, -2.32, -2.34, -2.39, -2.46, -2.54, -2.54, -2.71, -2.74, -2.84, -2.85, -2.93, -2.99, -3.08, -2.96, -3.06]
    # test_m_maxwr = [-1.64, -1.67, -1.72, -1.80, -1.82, -1.93, -1.95, -2.02, -2.10, -2.10, -2.14, -2.16, -2.23, -2.23, -2.25, -2.35, -2.49, -2.45, -2.47, -2.41, -2.48, -2.53, -2.58, -2.59]
    # test_l_maxwr = [-1.63, -1.72, -1.74, -1.79, -1.89, -2.02, -2.06, -2.14, -2.13, -2.21, -2.17, -2.20, -2.26, -2.31, -2.29, -2.36, -2.38, -2.42, -2.47, -2.50, -2.54, -2.64, -2.76, -2.75]

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Margin S=-1, M=-0.5, L=-0.25 test AP across scale: 
    # l_small = [21.70, 27.35, 30.71, 34.40, 36.67, 37.33, 38.56, 40.89, 40.29, 40.03, 42.37, 39.67, 41.93, 43.94, 41.37, 43.11, 42.38, 41.43, 44.19, 40.85, 40.48, 42.27, 41.15, 39.65]
    # l_midle = [40.72, 48.79, 53.30, 56.59, 58.27, 60.16, 61.69, 62.50, 63.31, 63.72, 64.62, 64.84, 65.58, 66.00, 66.38, 66.87, 67.13, 67.53, 67.60, 67.48, 67.52, 67.03, 67.08, 66.90]
    # l_large = [60.52, 69.38, 73.55, 76.82, 78.15, 79.67, 80.40, 81.26, 81.89, 82.42, 82.73, 83.20, 83.65, 83.84, 84.18, 84.42, 84.87, 85.00, 85.09, 85.44, 85.41, 85.38, 85.40, 85.55]

    # Margin S=-1, M=-0.5, L=-0.25 train correct & wrong avg margin
    # train_s_corrt = [2.36, 3.08, 3.42, 3.68, 3.87, 4.08, 4.23, 4.36, 4.51, 4.60, 4.74, 4.77, 4.87, 4.93, 5.08, 5.13, 5.29, 5.430, 5.510, 5.64, 5.750, 5.840, 5.970, 6.010]
    # train_m_corrt = [4.13, 5.21, 5.89, 6.41, 6.79, 7.04, 7.25, 7.46, 7.61, 7.77, 7.92, 8.08, 8.20, 8.28, 8.45, 8.55, 8.75, 8.790, 9.050, 9.18, 9.250, 9.410, 9.430, 9.440]
    # train_l_corrt = [4.75, 6.07, 6.84, 7.35, 7.75, 8.04, 8.32, 8.56, 8.71, 8.93, 9.07, 9.20, 9.40, 9.52, 9.74, 9.81, 9.91, 10.08, 10.27, 10.4, 10.48, 10.61, 10.66, 10.76]
    # train_s_maxwr = [-2.33, -2.32, -2.32, -2.34, -2.38, -2.32, -2.33, -2.35, -2.34, -2.36, -2.34, -2.38, -2.34, -2.36, -2.39, -2.35, -2.37, -2.39, -2.34, -2.38, -2.39, -2.38, -2.38, -2.34]
    # train_m_maxwr = [-2.02, -2.05, -2.09, -2.12, -2.12, -2.12, -2.14, -2.14, -2.17, -2.15, -2.14, -2.19, -2.20, -2.20, -2.19, -2.15, -2.15, -2.15, -2.19, -2.16, -2.20, -2.23, -2.14, -2.14]
    # train_l_maxwr = [-1.81, -1.85, -1.84, -1.89, -1.90, -1.91, -1.89, -1.90, -1.88, -1.90, -1.91, -1.86, -1.95, -1.94, -1.94, -1.94, -1.93, -1.92, -1.96, -1.98, -1.94, -2.02, -1.97, -1.98]

    # Margin S=-1, M=-0.5, L=-0.25 test correct & wrong avg margin
    # test_s_corrt = [3.27, 4.06, 4.63, 5.09, 5.39, 5.63, 5.85, 6.01, 6.18, 6.32, 6.44, 6.510, 6.680, 6.780, 6.890, 6.940, 7.130, 7.310, 7.430, 7.590, 7.680, 7.790, 7.940, 7.900]
    # test_m_corrt = [4.74, 5.99, 6.68, 7.31, 7.81, 8.11, 8.33, 8.57, 8.70, 8.91, 9.08, 9.140, 9.250, 9.380, 9.500, 9.600, 9.750, 9.830, 9.930, 10.08, 10.14, 10.25, 10.24, 10.25]
    # test_l_corrt = [5.63, 7.15, 7.78, 8.53, 8.92, 9.24, 9.49, 9.63, 9.72, 9.87, 9.98, 10.05, 10.14, 10.21, 10.38, 10.42, 10.42, 10.50, 10.56, 10.64, 10.71, 10.80, 10.74, 10.81]
    # test_s_maxwr = [-1.73, -1.78, -1.92, -1.98, -2.03, -2.10, -2.19, -2.28, -2.29, -2.38, -2.44, -2.58, -2.48, -2.63, -2.81, -2.75, -2.85, -2.88, -2.90, -2.97, -3.05, -3.06, -3.02, -3.06]
    # test_m_maxwr = [-1.71, -1.70, -1.74, -1.78, -1.88, -1.86, -1.92, -1.93, -2.05, -2.00, -2.05, -2.16, -2.27, -2.26, -2.31, -2.35, -2.32, -2.33, -2.30, -2.42, -2.50, -2.55, -2.52, -2.63]
    # test_l_maxwr = [-1.68, -1.77, -1.76, -1.81, -1.92, -1.94, -2.01, -2.08, -2.12, -2.22, -2.21, -2.23, -2.29, -2.35, -2.33, -2.40, -2.48, -2.48, -2.52, -2.59, -2.62, -2.60, -2.72, -2.84]

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Margin S=-1 test AP across scale: 
    # s_small = [21.96, 28.37, 30.40, 33.19, 34.64, 36.97, 37.62, 38.20, 38.84, 39.27, 40.55, 40.13, 41.60, 40.77, 41.44, 42.52, 42.27, 41.72, 39.72, 42.59, 43.10, 42.82, 42.89, 42.52]
    # s_midle = [42.42, 50.14, 54.11, 56.99, 59.50, 60.97, 62.19, 63.22, 64.34, 64.80, 65.10, 66.31, 66.96, 67.06, 67.09, 67.27, 67.88, 68.01, 67.79, 68.20, 67.97, 67.45, 67.96, 67.25]
    # s_large = [62.72, 69.65, 73.52, 76.24, 78.06, 79.61, 80.44, 81.33, 81.88, 82.42, 82.88, 83.25, 83.79, 83.90, 84.08, 84.44, 84.69, 84.96, 85.08, 85.16, 85.45, 85.29, 85.45, 85.36]

    # Margin S=-1, M=-0.5 test AP across scale: 
    # m_small = [24.11, 29.61, 32.55, 34.72, 36.24, 37.30, 37.24, 35.93, 38.22, 38.71, 39.93, 38.96, 39.55, 40.08, 40.89, 40.30, 41.82, 41.89, 42.44, 42.75, 42.62, 41.94, 41.62, 43.28]
    # m_midle = [42.79, 49.99, 54.11, 57.02, 59.24, 60.64, 61.62, 62.53, 63.49, 64.23, 64.49, 64.78, 65.28, 66.29, 66.37, 66.56, 67.08, 67.37, 67.54, 67.33, 67.11, 67.20, 67.63, 67.29]
    # m_large = [61.48, 69.39, 73.77, 76.76, 78.15, 79.43, 80.15, 81.03, 81.72, 82.28, 82.54, 82.84, 83.36, 83.82, 84.01, 84.56, 84.66, 84.68, 85.04, 85.28, 85.44, 85.43, 85.38, 85.52]
