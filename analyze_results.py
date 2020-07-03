#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: analyze_results
@time: 2020/6/8
@desc: 
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class AnalyzeResults(object):
    def __init__(self, table_path, sheet, index_col=0):
        """
        初始化
        :param table_path:
        :param sheet:
        :param index_col:
        """
        self.sheet = pd.read_excel(table_path, index_col=index_col, sheet_name=sheet)

    def plot_size_distribution(self, category, save_path):
        df = self.sheet
        df = df[df['category'] == category]
        size_lst = []
        for row in df.itertuples():
            adc_width = getattr(row, 'adc_width')
            adc_height = getattr(row, 'adc_height')
            size_lst.append(max(adc_width, adc_height))

        fig, ax = plt.subplots()
        fig.set_facecolor('papayawhip')

        size_lst = list(map(lambda x: min(110, x), size_lst))
        bins = list(range(0, 120, 10))

        n, bins, patches = ax.hist(x=size_lst, bins=bins, color='SkyBlue', edgecolor='k')
        plt.grid(axis='y', alpha=0.75)
        for index, num in enumerate(n):
            num = int(num)
            ax.annotate('{}'.format(num), xy=((bins[index] + bins[index + 1]) / 2, num),
                        xytext=(0, 0), textcoords="offset points",
                        ha='center', va='bottom')

        bins = np.delete(bins, -1)
        ax.set_xticks(bins)
        ax = plt.gca()
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(0)
        ax.set_xlabel('Size (μm)')
        ax.set_ylabel('Frequency')
        ax.set_title('<{}> Code size distribution'.format(category))
        plt.savefig(save_path, facecolor='papayawhip', bbox_inches='tight', dpi=300)
        print('FINISH.')

    def plot_score_distribution(self, category, save_path, score_thr=0.0):
        df = self.sheet
        df = df[df['category'] == category]
        score_lst = []
        for row in df.itertuples():
            score = getattr(row, 'score')
            img_name = getattr(row, 'image_name')
            if score < score_thr:
                print('[{}] score: {}'.format(img_name, score))
            score_lst.append(score)

        fig, ax = plt.subplots()
        fig.set_facecolor('papayawhip')

        bins = [x/10 for x in range(11)]

        n, bins, patches = ax.hist(x=score_lst, bins=bins, color='SkyBlue', edgecolor='k')
        plt.grid(axis='y', alpha=0.75)
        for index, num in enumerate(n):
            num = int(num)
            ax.annotate('{}'.format(num), xy=((bins[index] + bins[index + 1]) / 2, num),
                        xytext=(0, 0), textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_xticks(bins)
        ax = plt.gca()
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(0)
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('<{}> Code size distribution'.format(category))
        plt.savefig(save_path, facecolor='papayawhip', bbox_inches='tight', dpi=300)
        print('FINISH.')


if __name__ == '__main__':
    table_path = r'D:\Working\Tianma\13902\TEST\0609\test_1\deploy_results.xlsx'
    out_path = r'D:\Working\Tianma\13902\TEST\0609\test_1\analysis'
    os.makedirs(out_path, exist_ok=True)
    category = 'STR01'
    save_path = os.path.join(out_path, '{}_score_distribution.png'.format(category))

    analyze_table = AnalyzeResults(table_path, sheet='results')
    analyze_table.plot_score_distribution(category, save_path, score_thr=0.1)
