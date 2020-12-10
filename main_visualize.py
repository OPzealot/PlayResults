#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@contact: liusili@unionbigdata.com
@software:
@file: main_visualize
@time: 8/11/2020
@desc: 
"""
from visualize_results import visualize_results


def visualize_16902():
    sample_root = r'D:\Working\Tianma\16902\TEST\testset\testset_0729'
    results_table = r'D:\Working\Tianma\16902\TEST\results\out\deploy_results.xlsx'
    out_path = r'D:\Working\Tianma\16902\TEST\results\out\out'
    visualize_results(sample_root, results_table, out_path)


def visualize_54902():
    sample_root = r'D:\Working\Tianma\54902\TEST\testset\testset_check_0921'
    results_table = r'D:\Working\Tianma\54902\TEST\results\out\deploy_results.xlsx'
    out_path = r'D:\Working\Tianma\54902\TEST\results\out\out'
    visualize_results(sample_root, results_table, out_path)


def visualize_mask():
    sample_root = r'D:\Working\Tianma\Mask-FMM\TEST\data\test_1201'
    results_table = r'D:\Working\Tianma\Mask-FMM\TEST\result\out\deploy_results.xlsx'
    out_path = r'D:\Working\Tianma\Mask-FMM\TEST\result\out\out'
    visualize_results(sample_root, results_table, out_path)


def visualize_13902():
    sample_root = r'D:\Working\Tianma\13902\TEST\testset\13902_testset_0527'
    results_table = r'D:\Working\Tianma\13902\TEST\results\out\deploy_results.xlsx'
    out_path = r'D:\Working\Tianma\13902\TEST\results\out\out'
    visualize_results(sample_root, results_table, out_path)


def visualize_lighter():
    sample_root = r"E:\Working\Visionox\V2_lighter\TEST\data\testset_1106"
    results_table = r"E:\Working\Visionox\V2_lighter\TEST\results\out\deploy_results.xlsx"
    out_path = r'E:\Working\Visionox\V2_lighter\TEST\results\out\out'
    visualize_results(sample_root, results_table, out_path)


def visualize_lighter_large():
    sample_root = r"E:\Working\Visionox\V2_lighter\TEST\data\testset_large"
    results_table = r"E:\Working\Visionox\V2_lighter\TEST\results\out\deploy_results.xlsx"
    out_path = r'E:\Working\Visionox\V2_lighter\TEST\results\out\out'
    visualize_results(sample_root, results_table, out_path)


if __name__ == '__main__':
    visualize_mask()