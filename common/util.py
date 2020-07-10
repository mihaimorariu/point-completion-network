import numpy as np
import open3d as o3d
import torch

from pc_distance import tf_nndistance, tf_approxmatch


class ProgressMeter(object):
    def __init__(self, meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def set_prefix(self, prefix):
        self.prefix = prefix

    def display(self):
        entries = [self.prefix]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))


def read_cloud_points(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def save_cloud_points(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = torch.mean(torch.sqrt(dist1))
    dist2 = torch.mean(torch.sqrt(dist2))
    return (dist1 + dist2) / 2


# def mlp(features, layer_dims, bn=None, bn_params=None):
#     for i, num_outputs in enumerate(layer_dims[:-1]):
#         features = tf.contrib.layers.fully_connected(
#             features, num_outputs,
#             normalizer_fn=bn,
#             normalizer_params=bn_params,
#             scope='fc_%d' % i)
#     outputs = tf.contrib.layers.fully_connected(
#         features, layer_dims[-1],
#         activation_fn=None,
#         scope='fc_%d' % (len(layer_dims) - 1))
#     return outputs


# def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
#     for i, num_out_channel in enumerate(layer_dims[:-1]):
#         inputs = tf.contrib.layers.conv1d(
#             inputs, num_out_channel,
#             kernel_size=1,
#             normalizer_fn=bn,
#             normalizer_params=bn_params,
#             scope='conv_%d' % i)
#     outputs = tf.contrib.layers.conv1d(
#         inputs, layer_dims[-1],
#         kernel_size=1,
#         activation_fn=None,
#         scope='conv_%d' % (len(layer_dims) - 1))
#     return outputs


def point_maxpool(inputs, npts, keepdim=False):
    outputs = [f.max(dim=1, keepdim=keepdim) for f in torch.split(inputs, npts, dim=1)]
    return torch.cat(outputs, dim=0)


def point_unpool(inputs, npts):
    inputs = torch.split(inputs, inputs.shape[0], dim=0)
    outputs = [f.repeat(1, npts[i], 1) for i, f in enumerate(inputs)]
    return torch.cat(outputs, dim=1)


def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = pcd1.shape[1].to(np.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return torch.mean(cost / num_points)

# def add_train_summary(name, value):
# torch.utils.tensorboard.writer.SummaryWriter.add_scalar(name, value, collections=['train_summary'])


# def add_valid_summary(name, value):
# avg, update = torch.metrics.mean(value)
# tf.summary.scalar(name, avg, collections=['valid_summary'])
# return update
