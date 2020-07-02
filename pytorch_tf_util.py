import tensorflow as tf
import torch

from pc_distance import tf_nndistance, tf_approxmatch


class MLP(torch.nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        self.seqs = []

        for _, out_channels in enumerate(layer_dims):
            self.seqs.append(torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.ReLU()
            ))
            in_channels = out_channels

    def __call__(self, features):
        output = features
        for seq in self.seqs:
            output = seq(output)
        return output


class MLPConv(torch.nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(MLPConv, self).__init__()
        self.layer_dims = layer_dims
        self.conv1s = []

        for _, out_channels in enumerate(layer_dims):
            self.conv1s.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size=1))
            in_channels = out_channels

    def __call__(self, input):
        output = input
        for conv1 in self.conv1s:
            output = conv1(output)
        return output


def point_maxpool(input, npts, keepdim=False):
    split_size = [input.shape[2] // npt for npt in npts]
    output = [f.max(dim=2, keepdims=keepdim).values
              for f in torch.split(input, split_size, dim=2)]
    output = torch.cat(output, dim=0)
    return output


def point_unpool(input, npts):
    input = torch.split(input, input.shape[0], dim=0)
    output = [f.repeat(1, npts[i], 1) for i, f in enumerate(input)]
    output = torch.cat(output, dim=1)
    return output


def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = torch.mean(torch.sqrt(dist1))
    dist2 = torch.mean(torch.sqrt(dist2))
    return (dist1 + dist2) / 2


def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = pcd1.shape[1].to(torch.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return toch.mean(cost / num_points)


def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update
