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


class PlayResults(object):
    def __init__(self, result_path, test_json_path):
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
