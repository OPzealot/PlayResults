#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: visualize_results
@time: 2020/5/9
@desc: 
"""
import cv2
import numpy as np
import os
from ast import literal_eval
import pandas as pd
from tqdm import tqdm


def visualize_output(img_path, category, inference, bbox, score, **img_info):
    img = cv2.imread(img_path)
    height = img.shape[0]
    width = img.shape[1]
    font_size = round(height / 550, 1)
    font_length = int(font_size * 14)
    font_height = int(font_size * 20)
    thickness = int(font_size * 0.6 + 1)

    gt_text_cord = (1, font_height)
    cv2.putText(img, category, gt_text_cord,
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                font_size, (0, 0, 255), thickness)

    i = 1
    for key in img_info:
        color = (0, 0, 255)
        text = img_info[key]
        if key == 'short':
            short = img_info['short']
            if not np.isnan(short):
                if short == 0: text = '(0)No short.'
                elif short == 1: text = '(1)Short in pixel.'
                elif short == 2: text = '(2)Short between pixel.'
                color = (0, 255, 0)
            else: continue
        i += 1
        key_text_cord = (1, font_height * i)
        cv2.putText(img, key.upper() + ':{}'.format(text),
                    key_text_cord,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    font_size, color, thickness)

    infer_text_length = font_length * (len(inference + ':{:.3f}'.format(score)))
    infer_text_cord = (width - infer_text_length, font_height)
    cv2.putText(img, inference + ':{:.3f}'.format(score), infer_text_cord,
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                font_size, (0, 255, 0), thickness)
    if len(bbox) != 0:
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness)
        cv2.line(img, (xmax, ymin), infer_text_cord, (0, 255, 0), thickness)
    return img


def visualize_results(sample_root, results_table, out_path):
    results_df = pd.read_excel(results_table, index_col=0)
    pbar = tqdm(total=len(results_df))
    for row in results_df.itertuples():
        img_name = getattr(row, 'image_name')
        product = getattr(row, 'product')
        size = getattr(row, 'size')
        category = getattr(row, 'category')
        inference = getattr(row, 'inference')
        short = getattr(row, 'short')
        score = getattr(row, 'score')
        bbox = getattr(row, 'bbox')
        bbox = literal_eval(bbox)

        img_path = os.path.join(sample_root, category, img_name)
        new_dir = os.path.join(out_path, category, inference)
        os.makedirs(new_dir, exist_ok=True)
        new_img_path = os.path.join(new_dir, img_name)

        img = visualize_output(img_path, category, inference, bbox, score,
                               product=product, size=size, short=short)
        cv2.imwrite(new_img_path, img)
        pbar.set_description('Processing category [{}]'.format(category))
        pbar.update(1)
    print('[FINISH] Results have been visualized.')


if __name__ == '__main__':
    sample_root = r'D:\Working\Tianma\13902\TEST\13902_testset'
    results_table = r'D:\Working\Tianma\13902\TEST\0508\deploy_results.xlsx'
    out_path = r'D:\Working\Tianma\13902\TEST\0508\out'
    visualize_results(sample_root, results_table, out_path)
