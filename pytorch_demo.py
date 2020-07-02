import argparse
import numpy as np
import open3d as o3d
import torch

from common.visual import plot_side_by_side
from pytorch_models.pcn_cd import Model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='demo_data/car.pcd')
    parser.add_argument('--model_type', default='pcn_cd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_cd')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # device = torch.cuda.current_device()
    model = Model()
    # model.to(device)

    partial = o3d.io.read_point_cloud(args.input_path)
    partial = torch.Tensor(partial.points).transpose(1, 0)[None, :]
    complete = model(input=partial, npts=[partial.shape[0]])

    plot_side_by_side(partial, complete)
