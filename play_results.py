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
    def __init__(self, result_path, out_path, test_json_path=None, test_table_path=None, category_path=None):
        assert test_json_path is not None or test_table_path is not None
        self.out_path = out_path
        with open(result_path, 'rb') as f:
            self.results = pickle.load(f)

        self.test_json_path = test_json_path
        self.test_table_path = test_table_path
        if test_json_path is not None:
            self.categories = list(map(lambda x: x['name'], self.test_dict['categories']))
        else:
            assert category_path is not None
            categories = []
            for line in open(category_path, 'r'):
                lineTemp = line.strip()
                if lineTemp:
                    categories.append(lineTemp)
            self.categories = categories

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

    def get_gt_df_from_json(self):
        with open(self.test_json_path, 'r', encoding='utf-8') as f:
            test_dict = json.load(f)

        df = pd.DataFrame(np.zeros([len(self.results), len(self.categories)]),
                          index=range(len(self.results)), columns=self.categories)
        ann_lst = test_dict['annotations']
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

    def get_gt_df_from_table(self):
        table_df = pd.read_excel(self.test_table_path, index_col=0)

        df = pd.DataFrame(np.zeros([len(self.results), len(self.categories)]),
                          index=range(len(self.results)), columns=self.categories)

        for i in df.index:
            category = table_df.loc[i+1]['category']
            df.loc[i][category] = 1
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

        if self.test_json_path is not None:
            gt_df = self.get_gt_df_from_json()
        else:
            gt_df = self.get_gt_df_from_table()

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

            fig = plt.figure(figsize=(10, 8))
            ax = fig.subplots()
            fig.set_facecolor('papayawhip')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Precision')
            ax.set_xticks(thresh_lst)

            ax.set_title('[{}] Precision and Recall VS Threshold'.format(category))
            ax.grid(alpha=0.75, linestyle='--', color='y')
            ax.plot(thresh_lst, precision_lst, label='Precision')
            ax.plot(thresh_lst, recall_lst, label='Recall')
            ax.legend(bbox_to_anchor=(0.84, 0.01), loc=3, borderaxespad=0)

            save_path = os.path.join(thresh_path, category + '_by_thresh.png')
            plt.savefig(save_path, facecolor='papayawhip', bbox_inches='tight', dpi=300)
            pbar.set_description('Processing category {}'.format(category))
        print('Finished.')


if __name__ == '__main__':
    result_path = r'D:\Working\Tianma\13902\TEST\0508\deploy_results.pkl'
    # test_json_path = r'D:\Working\Tianma\13902\work_dir\test\test.json'
    out_path = r'D:\Working\Tianma\13902\TEST\0508'
    table_path = r'D:\Working\Tianma\13902\TEST\0508\deploy_results.xlsx'
    category_path = r'D:\Working\Tianma\13902\deploy\classes.txt'
    playResult = PlayResults(result_path, out_path,
                             test_json_path=None,
                             test_table_path=table_path,
                             category_path=category_path)
    playResult.pr_by_thresh()