import os

# from eval_voc_class import eval_voc_class
from eval_voc_scales_v1 import eval_voc_scale_v1


def class_evaluation(ckpt_path):
    mAP50 = eval_voc_class(ckpt_path)
    print(f"mAP50: {mAP50}")


def scale_evaluation(ckpt_path):
    margin, mAP50, scale_AP = eval_voc_scale_v1(ckpt_path)
    print(f"margin: {margin}")
    print(f"scale_AP: {scale_AP}")
    print(f"mAP50: {mAP50}")

if __name__ == '__main__':
    main_path = '/YOLOX_outputs'
    file_name = '/build_rewrite' # '/build_rewrite'
    wght_name = '/best_ckpt.pth' # '/best_ckpt.pth' # '/epoch_220_ckpt.pth'
    dataset = [('2007', 'test')] # 2012, trainval

    ckpt_path = os.getcwd() + main_path + file_name + wght_name
    scale_evaluation(ckpt_path)
