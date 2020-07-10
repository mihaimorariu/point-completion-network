import tensorflow as tf
import torch

from pc_distance import tf_nndistance, tf_approxmatch




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
