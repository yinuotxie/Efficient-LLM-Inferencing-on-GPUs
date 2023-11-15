# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: performance line chart

# !/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import optparse
import numpy as np
import matplotlib.pyplot as plt


def process_type(path):
    dim_idx = 0
    data_type = ''
    y_step = 1
    x_label = ''

    if 'prefill_seq' in path:
        dim_idx = 2
        data_type = 'prefill_seq'
        y_step = 10
        x_label = 'Seq Len'
    elif 'prefill_batch' in path:
        dim_idx = 1
        data_type = 'prefill_batch'
        y_step = 10
        x_label = 'Batch Size'
    elif 'decoding_seq' in path:
        dim_idx = 3
        data_type = 'decoding_seq'
        y_step = 1
        x_label = 'Seq Len'
    elif 'decoding_batch' in path:
        dim_idx = 1
        data_type = 'decoding_batch'
        y_step = 1
        x_label = 'Batch Size'
    elif 'hybrid' in path:
        dim_idx = -1
        data_type = 'hybrid'
        y_step = 10
        x_label = 'Prefill Fraction / %'
    else:
        raise TypeError('Unsupported data type')

    return dim_idx, data_type, y_step, x_label


def get_methods(log_file):
    methods = []

    with open(log_file) as fp:
        line = fp.readline()
        while line:
            if 'exit' in line and 'Naive' not in line:
                iterms = line.split(' ')
                methods.append(iterms[6])

            line = fp.readline()

    return methods


def get_dims(log_files, dim_idx):
    dims = []

    for log_file in log_files:
        dims.append(int((log_file.split('.')[0]).split('_')[dim_idx]))

    dims.sort()

    return dims


def read_data(methods, dims, data_path, log_files, dim_idx):
    data_throughput = np.zeros((len(methods), len(dims)), np.float64)
    data_performance = np.zeros((len(methods), len(dims)), np.float64)

    for log_file in log_files:
        dim = int((log_file.split('.')[0]).split('_')[dim_idx])
        with open(data_path + log_file) as fp:
            line = fp.readline()
            while line:
                if 'exit' in line and 'Naive' not in line:
                    iterms = line.split(' ')
                    method = iterms[6]
                    data_throughput[methods.index(
                        method)][dims.index(dim)] = float(iterms[14])
                    data_performance[methods.index(method)][dims.index(dim)] = float(
                        iterms[16].replace('(', '').replace(')', '').replace('%', ''))

                line = fp.readline()

    return data_throughput, data_performance


def draw_line_chart(methods, dims, data, figure_name, y_step, x_label, y_label, title):
    fig = plt.figure(figsize=(32, 24), dpi=100)

    dims_str = list(map(str, dims))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']

    for i in range(len(methods)):
        plt.plot(dims_str, data[i], color=colors[i % len(colors)],
                 linestyle=linestyles[(i // len(colors)) % len(linestyles)], marker='o', markersize=6)

    # plt.xticks(dims)
    plt.ylim(bottom=0)
    plt.yticks(
        range(0, round(np.max(np.max(data, axis=0)) + 0.5) + y_step, y_step))
    plt.tick_params(labelsize=25)

    # plt.hlines(y=100, xmin=dims_str[0], xmax=dims_str[-1], colors='r', linestyles='-.')
    plt.grid(True, linestyle='-.')

    plt.xlabel(x_label, fontdict={'size': '30'})
    plt.ylabel(y_label, fontdict={'size': '30'})
    plt.title(title, fontdict={'size': '30'})
    plt.legend(methods, loc='best', prop={'size': '30'})

    plt.savefig(figure_name, dpi=fig.dpi)
    # plt.show()


def analyze_data(data_path, dim_idx, data_type, y_step, x_label):
    log_files = []
    for file_name in os.listdir(data_path):
        if '.log' not in file_name:
            continue

        log_files.append(file_name)

    methods = get_methods(data_path + log_files[0])
    dims = get_dims(log_files, dim_idx)
    data_throughput, data_performance = read_data(
        methods, dims, data_path, log_files, dim_idx)
    draw_line_chart(methods, dims, data_throughput, data_path + data_type +
                    '_throughput.png', y_step, x_label, 'Throughput / TFLOPS', 'MHA Throughput')
    draw_line_chart(methods, dims, data_performance, data_path + data_type + '_performance.png', y_step, x_label,
                    'Performance Compared with Flash Attention / %', 'MHA Performance')


def main():
    usage = "python3 performance.py -p/--path log/"
    parser = optparse.OptionParser(usage)
    parser.add_option('-p', '--path', dest='path',
                      type='string', help='data path', default='log/')

    options, args = parser.parse_args()
    path = options.path

    dim_idx, data_type, y_step, x_label = process_type(path)
    analyze_data(path, dim_idx, data_type, y_step, x_label)


if __name__ == "__main__":
    main()
