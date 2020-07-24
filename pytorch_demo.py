import argparse

import numpy as np
import onnx
import open3d as o3d
import torch
from onnx import numpy_helper

from common.visual import plot_side_by_side


def torch2tf(torch_name):
    tf_name = torch_name
    tf_name = tf_name.replace('.inplace', '')
    tf_name = tf_name.replace('weight', 'weights')
    tf_name = tf_name.replace('bias', 'biases')
    tf_name = tf_name.replace('.', '/')
    tf_name = tf_name
    return tf_name


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=int, default=0, help='device id (-1 for cpu, non-zero value for gpu)')
    parser.add_argument('--input-file', '-i', type=str, default='demo_data/lamp.pcd', help='input (partial) point cloud file')
    parser.add_argument('--model-type', '-m', type=str, default='pcn_cd', help='model type', choices=['pcn_cd', 'pcn_emd', 'folding', 'fc'])
    parser.add_argument('--checkpoint', '-c', type=str, default='data/trained_models/pcn_cd', help='checkpoint file')
    parser.add_argument('--num-gt-points', '-p', type=int, default=16384, help='number of ground-truth points')
    parser.add_argument('--visualize', '-v', type=bool, default=False, help='visualize the reconstructed point cloud')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    if args.device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.device))

    if args.model_type == 'pcd_cd':
        from pytorch_models.pcn_cd import Model

    model = Model()

    onnx_model = onnx.load('model_tf.onnx')
    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph

    initalizers = dict()
    for init in graph.initializer:
        initalizers[init.name] = numpy_helper.to_array(init)

    for name, p in model.named_parameters():
        tf_name = torch2tf(name)
        dev_id = str(device).replace('cuda:', '') if args.device >= 0 else '0'
        p.data = (torch.from_numpy(initalizers[tf_name + ':' + dev_id])).data
        p.data = p.data.to(device)

        if tf_name.endswith('weights'):
            if len(p.data.shape) == 3:
                p.data = p.data.transpose(0, 2)
            if tf_name.find('fc_') != -1:
                p.data = p.data.transpose(0, 1)

    if args.device >= 0:
        model.to(device)

    partial = o3d.io.read_point_cloud(args.input_file)
    partial_numpy = np.asarray(partial.points, dtype=np.float32)[None, :]
    partial = torch.Tensor(partial.points).transpose(1, 0)[None, :]
    partial = partial.to(device)
    coarse, fine = model(partial)

    partial = partial.detach().cpu().numpy()[0].transpose(1, 0)
    fine = fine.detach().cpu().numpy()[0]
    coarse = coarse.detach().cpu().numpy()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fine)
    o3d.visualization.draw_geometries([pcd])

    plot_side_by_side(partial, fine)
