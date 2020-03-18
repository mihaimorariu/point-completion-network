#!/usr/bin/env python

import argparse
import open3d as o3d
import os
import numpy as np

from common.util import read_cloud_points, save_cloud_points
from common.visual import plot_side_by_side
from matplotlib import pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-m', '--model-type', type=str, default='pcn_cd')
    parser.add_argument('-c', '--checkpoint', type=str,
                        default='data/models/pcn_cd')
    parser.add_argument('-g', '--num-gt-points', type=int, default=16384)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    partial = read_cloud_points(args.input_path)
    complete = partial
    plot_side_by_side(partial, complete)

    if args.output_dir is None:
        visualize_cloud_points(complete)
        plt.show()
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(args.input_path))[0]

        output_file = os.path.join(args.output_dir, filename + '.pcd')
        save_cloud_points(output_file, complete)

        output_file = os.path.join(args.output_dir, filename + '.png')
        plt.savefig(output_file)


if __name__ == '__main__':
    main()
