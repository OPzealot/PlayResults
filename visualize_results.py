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
import colorsys
import numpy as np
import os
import sys
from ast import literal_eval
import pandas as pd
from tqdm import tqdm

COLOR = 4
# 点灯机
# FONTSIZE = 0.2
# 默认
FONTSIZE = 1.5
LABEL = 1


def get_color(inference, out_lst):
    """
    对不同框分配不同颜色（最多6种），特别：对最终预测框输出绿色
    :param inference:
    :param out_lst:
    :return:
    """
    global COLOR
    hsvList = [(x/COLOR, 0.8, 1.) for x in range(COLOR)]
    colorList = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvList))
    colorList = list(map(lambda x: [int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)], colorList))

    colorDict = {inference: [0, 255, 0]}
    i = 0
    for outInfo in out_lst:
        _, outCat, _ = outInfo
        if outCat not in colorDict:
            colorDict[outCat] = colorList[i]
            i = min(COLOR-1, i+1)

    return colorDict


def visualize_output(img_path, category, inference, score, bbox, out_lst, **img_info):
    global FONTSIZE
    global LABEL
    colorDict = get_color(inference, out_lst)
    if bbox and not out_lst or inference == 'Others':
        out_lst.append([bbox, inference, score])
    img = cv2.imread(img_path)
    height = img.shape[0]
    width = img.shape[1]
    fontScale = round(height / 750, 1) * FONTSIZE
    # for mask
    fontScale = round(height / 450, 1) * FONTSIZE
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    # fontScale = round(height / 250, 1)
    # font_length = int(fontScale * 14)
    # fontHeight = int(fontScale * 20)
    thickness = int(fontScale + 1)
    textSize, _ = cv2.getTextSize(category, fontFace, fontScale, thickness)
    fontHeight = int(textSize[1]*1.4)
    
    # gt_text_cord = (1, fontHeight)
    gt_text_cord = (1, height - 1)
    cv2.putText(img, category, gt_text_cord, fontFace, fontScale/FONTSIZE, (0, 0, 255), thickness)

    i = 1
    for key in img_info:
        color = (0, 0, 255)
        text = img_info[key]

        if not isinstance(text, (str, int, float, tuple)):
            continue

        if key == "adc_width":
            adc_width = img_info['adc_width']
            text = '{:.1f}um'.format(adc_width)
            color = (0, 255, 0)

        if key == "adc_height":
            adc_height = img_info['adc_height']
            text = '{:.1f}um'.format(adc_height)
            color = (0, 255, 0)

        if key == 'short':
            short = img_info['short']
            if not np.isnan(short):
                if short == 0:
                    text = '(0)No short.'
                elif short == 1:
                    text = '(1)Short in pixel.'
                elif short == 2:
                    text = '(2)Short between pixel.'
                color = (0, 255, 0)
            else:
                continue

        if key == 'defect_area':
            area = img_info['defect_area']
            text = '{:.1f} pixels'.format(area)
            color = (0, 255, 255)

        if key == 'ab_info':
            ab_info = img_info['ab_info']
            ab_lst, percentage_lst = ab_info
            for i, ab in enumerate(ab_lst):
                percentage = percentage_lst[i]
                xmin = max(int(ab[0]), 0)
                ymin = max(int(ab[1]), 0)
                xmax = min(int(ab[2]), width)
                ymax = min(int(ab[3]), height)

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness)

                abText = '{:.0f}%'.format(percentage*100)
                abTextSize, _ = cv2.getTextSize(abText, fontFace, fontScale, thickness)
                abCord = (min(xmin, width - abTextSize[0]), ymin + fontHeight)
                cv2.putText(img, abText, abCord, fontFace, fontScale, (0, 0, 255), thickness)
            continue

        i += 1
        key_text_cord = (1, fontHeight * i)
        cv2.putText(img, key.upper() + ':{}'.format(text),
                    key_text_cord, fontFace, fontScale, color, thickness)
    
    inferText = inference + ':{:.3f}'.format(score)
    inferTextSize, _ = cv2.getTextSize(inferText, fontFace, fontScale, thickness)
    inferTextCord = (width - inferTextSize[0], fontHeight)
    # cv2.putText(img, inference + ':{:.3f}'.format(score), inferTextCord,
    #             fontFace, fontScale, (0, 255, 0), thickness)

    fontScale = fontScale * 0.6
    for outInfo in out_lst:
        bbox, outCat, outScore = outInfo
        color = colorDict[outCat]
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        outText = outCat + ':' + str(round(outScore, 2))

        outTextSize, _ = cv2.getTextSize(outText, fontFace, fontScale, thickness)

        if ymin > outTextSize[1]:
            topLeft = (xmin-thickness+1, ymin-outTextSize[1]-3)
            bottomRight = (xmin+outTextSize[0], ymin)
            outTextCord = (xmin, ymin - 2)
        else:
            topLeft = (xmin-thickness+1, ymax-thickness+1)
            bottomRight = (xmin+outTextSize[0], ymax+outTextSize[1]+3)
            outTextCord = (xmin, ymax+outTextSize[1])

        cv2.rectangle(img, topLeft, bottomRight, color, -1)
        cv2.putText(img, outText, outTextCord, fontFace, fontScale, (0, 0, 0), LABEL)

    return img


def visualize_results(sample_root, results_table, out_path):
    results_df = pd.read_excel(results_table, index_col=0)
    pbar = tqdm(total=len(results_df), file=sys.stdout)
    for row in results_df.itertuples():
        img_name = getattr(row, 'image_name')
        product = getattr(row, 'product', None)
        size = getattr(row, 'size', None)
        category = getattr(row, 'category')
        category = str(category)
        inference = getattr(row, 'inference')
        short = getattr(row, 'short', None)
        score = getattr(row, 'score')
        bbox = getattr(row, 'bbox')
        bbox = literal_eval(bbox)
        defect_area = getattr(row, 'defect_area', None)
        ab_info = getattr(row, 'ab_info', None)
        if ab_info:
            ab_info = literal_eval(ab_info)
        adc_width = getattr(row, 'adc_width', None)
        adc_height = getattr(row, 'adc_height', None)

        out_lst = getattr(row, 'output', '[]')
        out_lst = literal_eval(out_lst)
        # op1 = getattr(row, 'OP1')
        # op2 = getattr(row, 'OP2')
        # op3 = getattr(row, 'OP3')
        # engineer = getattr(row, 'engineer')

        img_path = os.path.join(sample_root, category, img_name)
        new_dir = os.path.join(out_path, category, inference)
        os.makedirs(new_dir, exist_ok=True)
        new_img_path = os.path.join(new_dir, img_name)

        img = visualize_output(img_path, category, inference, score, bbox, out_lst,
                               product=product, size=size, ab_info=ab_info,
                               adc_width=adc_width, adc_height=adc_height, short=short,
                               defect_area=defect_area)

        # tqdm.write('[CATE]:{}, [IMAGE]:{}'.format(category, img_name))
        # img = visualize_output(img_path, category, inference, bbox, score, ab_info=ab_info)

        cv2.imwrite(new_img_path, img)
        pbar.set_description('Processing category [{}]'.format(category))
        pbar.update(1)
    print('[FINISH] Results have been visualized.')


