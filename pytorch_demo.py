import argparse

import numpy as np
import onnx
import open3d as o3d
import torch
import onnxruntime as ort

from common.visual import plot_side_by_side
from pytorch_models.pcn_cd import Model
from onnx import numpy_helper

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
    parser.add_argument('--input_path', default='demo_data/lamp.pcd')
    parser.add_argument('--model_type', default='pcn_cd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_cd')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    device = torch.cuda.current_device()
    model = Model()
    # model.to(device)
    # raise

    onnx_model = onnx.load('model_tf.onnx')
    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph

    initalizers = dict()
    for init in graph.initializer:
        initalizers[init.name] = numpy_helper.to_array(init)

    for name, p in model.named_parameters():
        tf_name = torch2tf(name)
        p.data = (torch.from_numpy(initalizers[tf_name + ':' + str(device)])).data

        if tf_name.endswith('weights'):
            if len(p.data.shape) == 3:
                p.data = p.data.transpose(0, 2)
            if tf_name.find('fc_') != -1:
                p.data = p.data.transpose(0, 1)

    partial = o3d.io.read_point_cloud(args.input_path)
    partial_numpy = np.asarray(partial.points, dtype=np.float32)[None, :]
    partial = torch.Tensor(partial.points).transpose(1, 0)[None, :]
    coarse, fine = model(partial)

    partial = partial.detach().numpy()[0].transpose(1, 0)
    fine = fine.detach().numpy()[0]
    coarse = coarse.detach().numpy()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fine)
    o3d.visualization.draw_geometries([pcd])

    plot_side_by_side(partial, fine)
