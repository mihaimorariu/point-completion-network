import torch
from torch.nn.functional import relu
from common.helper import point_unpool, point_maxpool


class Folding(torch.nn.Module):
    def __init__(self, in_channels, layer_dims, grid_size, grid_scale, num_coarse, num_fine):
        super(Folding, self).__init__()
        self.layer_dims = layer_dims
        self.grid_size = grid_size
        self.grid_scale = grid_scale
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        for i, out_channels in enumerate(self.layer_dims):
            layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            setattr(self, 'conv_' + str(i), layer)
            in_channels = out_channels

        x = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        y = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)

        self.grid = torch.meshgrid(x, y)
        self.grid = torch.stack(self.grid, axis=2)
        self.grid = self.grid.transpose(1, 0)
        self.grid = torch.reshape(self.grid, [-1, 2])
        self.grid = torch.unsqueeze(self.grid, 0)

    def __call__(self, features, coarse):
        grid_feat = self.grid.repeat(features.shape[0], self.num_coarse, 1)
        grid_feat = grid_feat.to(features.device)

        point_feat = torch.unsqueeze(coarse, 2)
        point_feat = point_feat.repeat(1, 1, self.grid_size ** 2, 1)
        point_feat = torch.reshape(point_feat, [-1, self.num_fine, 3])

        global_feat = torch.unsqueeze(features, 1)
        global_feat = global_feat.repeat(1, self.num_fine, 1)

        feat = torch.cat([grid_feat, point_feat, global_feat], axis=2)

        center = torch.unsqueeze(coarse, 2)
        center = center.repeat(1, 1, self.grid_size ** 2, 1)
        center = torch.reshape(center, [-1, self.num_fine, 3])

        fine = feat.transpose(1, 2)
        dims = len(self.layer_dims)
        for i in range(dims):
            layer = getattr(self, 'conv_' + str(i))
            if i == dims - 1:
                fine = layer(fine)
            else:
                fine = relu(layer(fine))

        fine = fine.transpose(1, 2) + center
        return fine


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(Encoder, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(self.layer_dims):
            layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            setattr(self, 'conv_' + str(i), layer)
            in_channels = out_channels

    def __call__(self, inputs):
        outputs = inputs
        dims = len(self.layer_dims)
        for i in range(dims):
            layer = getattr(self, 'conv_' + str(i))
            if i == dims - 1:
                outputs = layer(outputs)
            else:
                outputs = relu(layer(outputs))
        return outputs


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(Decoder, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(layer_dims):
            layer = torch.nn.Linear(in_channels, out_channels)
            setattr(self, 'fc_' + str(i), layer)
            in_channels = out_channels

    def __call__(self, inputs):
        outputs = inputs
        dims = len(self.layer_dims)
        for i in range(dims):
            layer = getattr(self, 'fc_' + str(i))
            if i == dims - 1:
                outputs = layer(outputs)
            else:
                outputs = relu(layer(outputs))
        return outputs


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse

        self.encoder_0 = Encoder(in_channels=3, layer_dims=[128, 256])
        self.encoder_1 = Encoder(in_channels=512, layer_dims=[512, 1024])
        self.decoder = Decoder(in_channels=1024, layer_dims=[1024, 1024, self.num_coarse * 3])
        self.folding = Folding(in_channels=1029, layer_dims=[512, 512, 3], grid_size=self.grid_size, grid_scale=self.grid_scale,
                               num_coarse=self.num_coarse, num_fine=self.num_fine)

    def __call__(self, inputs):
        features = self.encode(inputs)
        coarse, fine = self.decode(features)
        return coarse, fine

    def encode(self, inputs):
        num_points = [inputs.shape[2]]
        features = self.encoder_0(inputs)
        features_global = point_maxpool(features, num_points, keepdim=True)
        features_global = point_unpool(features_global, num_points)
        features = torch.cat([features, features_global], dim=1)
        features = self.encoder_1(features)
        features = point_maxpool(features, num_points)
        return features

    def decode(self, features):
        coarse = self.decoder(features)
        coarse = torch.reshape(coarse, [-1, self.num_coarse, 3])
        fine = self.folding(features, coarse)
        return coarse, fine
