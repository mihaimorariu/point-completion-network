from pytorch_tf_util import *


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse

        self.mlp_conv1 = MLPConv(in_channels=3, layer_dims=[128, 256])
        self.mlp_conv2 = MLPConv(in_channels=512, layer_dims=[512, 1024])
        self.mlp_conv3 = MLPConv(in_channels=1029, layer_dims=[512, 512, 3])
        self.mlp = MLP(in_channels=1024, layer_dims=[1024, 1024, self.num_coarse * 3])

    def __call__(self, input, npts, gt=None, alpha=None):
        features = self.encode(input, npts)
        coarse, fine = self.decode(features)

        if gt is not None:
            assert alpha is not None
            loss = self.loss(coarse, fine, gt, alpha)
            return loss

        return coarse, fine

    def encode(self, inputs, npts):
        features = self.mlp_conv1(inputs)
        features_global = point_unpool(point_maxpool(features, npts, keepdim=True), npts)
        features_global = features_global.repeat(1, 1, features.shape[2])
        features = torch.cat([features, features_global], dim=1)
        features = self.mlp_conv2(features)
        features = point_maxpool(features, npts)
        return features

    def decode(self, features):
        coarse = self.mlp(features)
        coarse = torch.reshape(coarse, [-1, self.num_coarse, 3])

        x = torch.linspace(-0.05, 0.05, self.grid_size)
        y = torch.linspace(-0.05, 0.05, self.grid_size)

        grid = torch.meshgrid(x, y)
        grid = torch.unsqueeze(torch.reshape(torch.stack(grid, axis=2), [-1, 2]), 0)
        grid_feat = grid.repeat(features.shape[0], self.num_coarse, 1)

        point_feat = torch.unsqueeze(coarse, 2)
        point_feat = point_feat.repeat(1, 1, self.grid_size ** 2, 1)
        point_feat = torch.reshape(point_feat, [-1, self.num_fine, 3])

        global_feat = torch.unsqueeze(features, 1)
        global_feat = global_feat.repeat(1, self.num_fine, 1)

        feat = torch.cat([grid_feat, point_feat, global_feat], axis=2)

        center = torch.unsqueeze(coarse, 2)
        center = center.repeat(1, 1, self.grid_size ** 2, 1)
        center = torch.reshape(center, [-1, 3, self.num_fine])

        feat = feat.transpose(1, 2)
        fine = self.mlp_conv3(feat)
        fine = fine + center

        return coarse, fine

    def loss(self, coarse, fine, gt, alpha):
        loss_coarse = chamfer(coarse, gt)
    #     add_train_summary('train/coarse_loss', loss_coarse)
    #     update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)
    #
        loss_fine = chamfer(fine, gt)
    #     add_train_summary('train/fine_loss', loss_fine)
    #     update_fine = add_valid_summary('valid/fine_loss', loss_fine)
    #
        loss = loss_coarse + alpha * loss_fine
    #     add_train_summary('train/loss', loss)
    #     update_loss = add_valid_summary('valid/loss', loss)
    #
        return loss
