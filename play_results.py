#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: play_results
@time: 2020/4/20
@desc: 
"""
import pickle
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm


class PlayResults(object):
    def __init__(self, result_path, test_json_path, out_path):
        self.out_path = out_path
        with open(result_path, 'rb') as f:
            self.results = pickle.load(f)
        with open(test_json_path, 'r', encoding='utf-8') as f:
            self.test_dict = json.load(f)
        self.categories = list(map(lambda x: x['name'], self.test_dict['categories']))

    def get_prediction_df(self):
        df = pd.DataFrame(np.zeros([len(self.results), len(self.categories)]),
                          index=range(len(self.results)), columns=self.categories)

        for img_index in range(len(self.results)):
            for cat_id, cat in enumerate(self.results[img_index]):
                for bbox in cat:
                    conf = bbox[4]
                    if conf > df.iloc[img_index, cat_id]:
                        df.iloc[img_index, cat_id] = conf
        return df

    def get_gt_df(self):
        df = pd.DataFrame(np.zeros([len(self.results), len(self.categories)]),
                          index=range(len(self.results)), columns=self.categories)
        ann_lst = self.test_dict['annotations']
        img_id = None
        img_index = -1
        for i in range(len(ann_lst)):
            tmp_id = ann_lst[i]['image_id']
            if img_id is None or img_id != tmp_id:
                img_id = tmp_id
                img_index += 1
            cat_id = ann_lst[i]['category_id'] - 1
            df.iloc[img_index, cat_id] = 1
        return df

    @staticmethod
    def get_precision_recall(gt, predict, thresh):
        gt_total = gt.sum()
        predict = predict > thresh
        predict_total = predict.sum()
        correct = (predict * gt).astype(bool).sum()
        precision = round(correct / predict_total, 3)
        recall = round(correct / gt_total, 3)
        return precision, recall

    def pr_by_thresh(self):
        thresh_path = os.path.join(self.out_path, 'Thresh')
        os.makedirs(thresh_path, exist_ok=True)
        gt_df = self.get_gt_df()
        predict_df = self.get_prediction_df()

        pbar = tqdm(self.categories)
        for category in pbar:
            gt = gt_df[category]
            predict = predict_df[category]
            thresh_lst = np.arange(0, 1.0, 0.05)
            precision_lst = []
            recall_lst = []

            for thresh in thresh_lst:
                precision, recall = self.get_precision_recall(gt, predict, thresh)
                precision_lst.append(precision)
                recall_lst.append(recall)

            fig = plt.figure(figsize=(20, 8))
            ax_1, ax_2 = fig.subplots(1, 2)
            fig.set_facecolor('papayawhip')
            ax_1.set_xlabel('Threshold')
            ax_1.set_ylabel('Precision')
            ax_1.set_xticks(thresh_lst)

            ax_1.set_title('[{}] Precision VS Threshold'.format(category))
            ax_1.grid(alpha=0.75, linestyle='--', color='y')
            ax_1.plot(thresh_lst, precision_lst)

            ax_2.set_xlabel('Threshold')
            ax_2.set_ylabel('Recall')
            ax_2.set_xticks(thresh_lst)
            ax_2.set_title('[{}] Recall VS Threshold'.format(category))
            ax_2.grid(alpha=0.75, linestyle='--', color='y')
            ax_2.plot(thresh_lst, recall_lst)

            save_path = os.path.join(thresh_path, category + '_by_thresh.png')
            plt.savefig(save_path, facecolor='papayawhip', bbox_inches='tight', dpi=300)
            pbar.set_description('Processing category {}'.format(category))
        print('Finished.')

if __name__ == '__main__':
    result_path = r'D:\Working\Tianma\18902\work_dir\2020_0416\model.pth.pkl'
    test_json_path = r'D:\Working\Tianma\18902\work_dir\2020_0416\test.json'
    out_path = r'D:\Working\Tianma\18902\work_dir\2020_0416'
    playResult = PlayResults(result_path, test_json_path, out_path)
    playResult.pr_by_thresh()