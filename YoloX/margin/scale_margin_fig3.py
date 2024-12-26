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
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True) # figure

    # setting up font
    font = {'family' : 'Times New Roman', 'size' : 12}
    plt.rc('font', **font)
    
    color16 = "#D2AA3A" # 科研黄
    color45 = "#D94738" # 科研红
    color55 = "#039F89" # 科研绿
    color61 = "#1C6AB1" # 科研蓝
    color77 = "#A04294" # 科研紫
    color88 = "#F7C2CD" # 科研粉

    color_list = [color16, color16, color45, color45, color55, color55, color77]

    # setting up background & lines
    # sns.set_style("whitegrid") # 设置背景样式
    # for i in range(len(data_y)):
    #     if i == 0 or i == 2 or i == 4:
    #         sns.lineplot(x=data_x, y=data_y[i], color=color_list[i], linewidth=1, marker="o", markersize=1, markeredgecolor=color_list[i], markeredgewidth=2, label=data_label[i])
    #     if i == 1 or i == 3:
    #         sns.lineplot(x=data_x, y=data_y[i], color=color_list[i], linewidth=1, linestyle='--', marker="D", markersize=1, markeredgecolor=color_list[i], markeredgewidth=2, label=data_label[i])
    
    # 设置边框和网格
    for ax in axs:
        ax.spines['top'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['right'].set_color('gray')
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)  # 设置灰色网格

    # 绘制第一个子图
    axs[0].plot(data_x, data_y[0], color77, linewidth=2, marker="o", markersize=1, markeredgecolor=color77, markeredgewidth=2, label="$\delta_{s}=0, \delta_{m}=0, \delta_{l}=0$")
    axs[0].plot(data_x, data_y[1], color16, linewidth=2, marker="D", markersize=1, markeredgecolor=color16, markeredgewidth=2, label="$\delta_{s}=1, \delta_{m}=0, \delta_{l}=0$")
    axs[0].axvline(x=210, color=color16, linestyle='--', linewidth=1.5, label='$Best ~ Generalization$')
    axs[0].set_ylim(20, 45)
    axs[0].set_xlim(0, 250)
    axs[0].legend(loc='lower right', frameon=True, fontsize=12)

    # 绘制第二个子图
    axs[1].plot(data_x, data_y[0], color77, linewidth=2, marker="o", markersize=1, markeredgecolor=color77, markeredgewidth=2, label="$\delta_{s}=0, \delta_{m}=0, \delta_{l}=0$")
    axs[1].plot(data_x, data_y[2], color45, linewidth=2, marker="D", markersize=1, markeredgecolor=color45, markeredgewidth=2, label=r"$\delta_{s}=1, \delta_{m}=\frac{1}{2}, \delta_{l}=0$")
    axs[1].axvline(x=200, color=color45, linestyle='--', linewidth=1.5, label='$Best ~ Generalization$')
    axs[1].set_ylim(20, 45)
    axs[1].set_xlim(0, 250)
    axs[1].legend(loc='lower right', frameon=True, fontsize=12)

    # 绘制第三个子图
    axs[2].plot(data_x, data_y[0], color77, linewidth=2, marker="o", markersize=1, markeredgecolor=color77, markeredgewidth=2, label="$\delta_{s}=0, \delta_{m}=0, \delta_{l}=0$")
    axs[2].plot(data_x, data_y[3], color55, linewidth=2, marker="D", markersize=1, markeredgecolor=color55, markeredgewidth=2, label=r"$\delta_{s}=1, \delta_{m}=\frac{1}{2}, \delta_{l}=\frac{1}{4}$")
    axs[2].axvline(x=190, color=color55, linestyle='--', linewidth=1.5, label='$Best ~ Generalization$')
    axs[2].set_ylim(20, 45)
    axs[2].set_xlim(0, 250)
    axs[2].legend(loc='lower right', frameon=True, fontsize=12)
    
    # 设置 x 轴标签
    axs[2].set_xlabel('Epoch', fontsize=14)

    # 设置 y 轴标签
    axs[1].set_ylabel('$AP_{s}$', fontsize=14, labelpad=12)

    # 设置总标题
    axs[0].set_title(title, fontweight='bold', fontsize=14)

    # 最佳泛化点竖线
    # plt.axvline(x=240, color=color77, linestyle='--', linewidth=1, label='$Overall ~ Best ~ Generalization$')

    # setting up title
    # plt.title(title, fontweight='bold', fontsize=12)
    # plt.xlabel(x_label, fontsize=10)
    # plt.ylabel(y_label, fontsize=10)
    
    # setting up label
    # plt.legend(loc='lower right', frameon=True, fontsize=10)

    # 设置刻度字体和范围
    # data_x = [50, 100, 150, 200, 240]
    # plt.xticks(ticks=data_x, fontsize=8)
    # plt.yticks(fontsize=8)
    # plt.xlim(x_range[0], x_range[1])
    # plt.ylim(y_range[0], y_range[1])
    
    # 设置坐标轴样式
    # for spine in plt.gca().spines.values():
    #     spine.set_edgecolor("#CCCCCC")
    #     spine.set_linewidth(1.5)

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

    title = 'Test Set Small Object Average Precision'

    x_range = [0, 250] # 
    y_range = [0, 60] # 

    x_label, y_label = 'Epoch', 'Average Precision'

    data_label = ['$Baseline ~ Small ~ Object ~ AP_{s}$', 
                  '$\delta_{s}=1 ~ Small ~ Object ~ AP_{s}$', 
                  '$\delta_{s}=1, \delta_{m}=0.5 ~ Small ~ Object ~ AP_{s}$', 
                  '$\delta_{s}=1, \delta_{m}=0.5, \delta_{m}=0.25 ~ Small ~ Object ~ AP_{s}$',]
    
    save_path = '/home/uic/ChengYuxuan/YoloXDSTRe1/tblogger/' + 'comparison' # test_accuracy # train_margin

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

    # baseline test AP across scale: 
    bl_small = [20.64, 27.49, 30.63, 34.47, 36.07, 36.63, 38.76, 38.88, 40.74, 40.71, 39.53, 41.23, 40.46, 39.97, 41.02, 41.58, 41.42, 42.18, 41.26, 40.82, 41.46, 41.25, 40.74, 40.67]
    bl_midle = [40.55, 49.26, 53.13, 56.77, 59.05, 60.70, 61.84, 62.42, 63.46, 64.11, 63.92, 65.36, 65.49, 65.90, 66.36, 66.81, 66.77, 67.23, 67.55, 67.52, 66.22, 66.69, 66.78, 66.51]
    bl_large = [59.71, 69.28, 73.54, 76.72, 78.34, 80.01, 80.70, 81.03, 81.66, 82.43, 82.43, 83.22, 83.46, 83.70, 84.11, 84.71, 84.50, 84.83, 84.97, 85.10, 85.23, 85.32, 85.42, 85.42]

    # Margin S=-1 test AP across scale: 
    s_small = [21.96, 28.37, 30.40, 33.19, 34.64, 36.97, 37.62, 38.20, 38.84, 39.27, 40.55, 40.13, 41.60, 40.77, 41.44, 42.52, 42.27, 41.72, 39.72, 42.59, 43.10, 42.82, 42.89, 42.52]
    s_midle = [42.42, 50.14, 54.11, 56.99, 59.50, 60.97, 62.19, 63.22, 64.34, 64.80, 65.10, 66.31, 66.96, 67.06, 67.09, 67.27, 67.88, 68.01, 67.79, 68.20, 67.97, 67.45, 67.96, 67.25]
    s_large = [62.72, 69.65, 73.52, 76.24, 78.06, 79.61, 80.44, 81.33, 81.88, 82.42, 82.88, 83.25, 83.79, 83.90, 84.08, 84.44, 84.69, 84.96, 85.08, 85.16, 85.45, 85.29, 85.45, 85.36]

    # Margin S=-1, M=-0.5 test AP across scale: 
    m_small = [24.11, 29.61, 32.55, 34.72, 36.24, 37.30, 37.24, 35.93, 38.22, 38.71, 39.93, 38.96, 39.55, 40.08, 40.89, 40.30, 41.82, 41.89, 42.44, 42.75, 42.62, 41.94, 41.62, 43.28]
    m_midle = [42.79, 49.99, 54.11, 57.02, 59.24, 60.64, 61.62, 62.53, 63.49, 64.23, 64.49, 64.78, 65.28, 66.29, 66.37, 66.56, 67.08, 67.37, 67.54, 67.33, 67.11, 67.20, 67.63, 67.29]
    m_large = [61.48, 69.39, 73.77, 76.76, 78.15, 79.43, 80.15, 81.03, 81.72, 82.28, 82.54, 82.84, 83.36, 83.82, 84.01, 84.56, 84.66, 84.68, 85.04, 85.28, 85.44, 85.43, 85.38, 85.52]

    # Margin S=-1, M=-0.5, L=-0.25 test AP across scale: 
    l_small = [21.70, 27.35, 30.71, 34.40, 36.67, 37.33, 38.56, 40.89, 40.29, 40.03, 42.37, 39.67, 41.93, 43.94, 41.37, 43.11, 42.38, 41.43, 44.19, 40.85, 40.48, 42.27, 41.15, 39.65]
    l_midle = [40.72, 48.79, 53.30, 56.59, 58.27, 60.16, 61.69, 62.50, 63.31, 63.72, 64.62, 64.84, 65.58, 66.00, 66.38, 66.87, 67.13, 67.53, 67.60, 67.48, 67.52, 67.03, 67.08, 66.90]
    l_large = [60.52, 69.38, 73.55, 76.82, 78.15, 79.67, 80.40, 81.26, 81.89, 82.42, 82.73, 83.20, 83.65, 83.84, 84.18, 84.42, 84.87, 85.00, 85.09, 85.44, 85.41, 85.38, 85.40, 85.55]
    
    visualize_control([bl_small, s_small, m_small, l_small]) # s_small, m_small, l_small

    # baseline train margin
    # small = [20.64, 27.49, 30.83, 34.47, 36.07, 36.63, 38.76, 38.88, 40.74, 40.71, 39.53, 41.23, 40.46, 39.97, 41.02, 41.58, 41.42, 42.18, 41.26, 40.82, 41.46, 41.25, 40.74, 40.67]
    # midle = [40.55, 49.26, 53.13, 56.77, 59.05, 60.70, 61.84, 62.42, 63.46, 64.11, 63.92, 65.36, 65.49, 65.90, 66.36, 66.81, 66.77, 67.23, 67.55, 67.52, 66.22, 66.69, 66.78, 66.51]
    # large = [59.71, 69.28, 73.54, 76.72, 78.34, 80.01, 80.70, 81.03, 81.66, 82.43, 82.43, 83.22, 83.46, 83.70, 84.11, 84.71, 84.50, 84.83, 84.97, 85.10, 85.23, 85.32, 85.42, 85.42]
    
    # baseline test margin
    # small = [20.64, 27.49, 30.63, 34.47, 36.07, 36.63, 38.76, 38.88, 40.74, 40.71, 39.53, 41.23, 40.46, 39.97, 41.02, 41.58, 41.42, 42.18, 41.26, 40.82, 41.46, 41.25, 40.74, 42.18]
    # midle = [40.55, 49.26, 53.13, 56.77, 59.05, 60.70, 61.84, 62.42, 63.46, 64.11, 63.92, 65.36, 65.49, 65.90, 66.36, 66.81, 66.77, 67.23, 67.55, 67.52, 66.22, 66.69, 66.78, 66.51]
    # large = [59.71, 69.28, 73.54, 76.72, 78.34, 80.01, 80.70, 81.03, 81.66, 82.43, 82.43, 83.22, 83.46, 83.70, 84.11, 84.71, 84.50, 84.83, 84.97, 85.10, 85.23, 85.32, 85.42, 85.42]

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    

    
