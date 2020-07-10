from pytorch_tf_util import *


class Folding(torch.nn.Module):
    def __init__(self, in_channels, layer_dims, grid_size):
        super(Folding, self).__init__()
        self.layer_dims = layer_dims
        self.grid_size = grid_size

        for i, out_channels in enumerate(self.layer_dims):
            layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            setattr(self, 'conv_' + str(i), layer)
            in_channels = out_channels

        x = torch.linspace(-0.05, 0.05, self.grid_size)
        y = torch.linspace(-0.05, 0.05, self.grid_size)

        self.meshgrid = torch.meshgrid(x, y)
        self.meshgrid = torch.unsqueeze(torch.reshape(torch.stack(self.meshgrid, axis=2), [-1, 2]), 0)

    def __call__(self, features, coarse):
        grid_feat = self.meshgrid.repeat(features.shape[0], self.num_coarse, 1)

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
        fine = feat

        for i in range(len(self.layer_dims)):
            layer = getattr(self, 'conv_' + str(i))
            fine = layer(fine)

        fine = fine + center
        return fine


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(Encoder, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(self.layer_dims):
            layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            setattr(self, 'conv_' + str(i), layer)
            in_channels = out_channels

    def __call__(self):
        output = input
        for i in range(len(self.layer_dims)):
            layer = getattr(self, 'conv_' + str(i))
            output = layer(output)
        return output


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, layer_dims):
        super(Decoder, self).__init__()
        self.layer_dims = layer_dims
        for i, out_channels in enumerate(layer_dims):
            layer = torch.nn.ReLU(torch.nn.Linear(in_channels, out_channels))
            setattr(self, 'fc_' + str(i), layer)
            in_channels = out_channels

    def __call__(self):
        output = input
        for i in range(len(self.layer_dims)):
            layer = getattr(self, 'fc_' + str(i))
            output = layer(output)
        return output


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
        self.folding = Folding(in_channels=1029, layer_dims=[1024, 1024, self.num_coarse * 3], grid_size=self.grid_size)

    def __call__(self, input):
        features = self.encode(input)
        coarse, fine = self.decode(features)
        return coarse, fine

    def encode(self, input):
        num_points = input.shape[1]
        features = self.encoder_0(input)
        features_global = point_maxpool(features, num_points, keepdim=True)
        features_global = point_unpool(features_global, num_points)
        features_global = features_global.repeat(1, 1, features.shape[2])
        features = torch.cat([features, features_global], dim=1)
        features = self.encoder_1(features)
        features = point_maxpool(features, num_points)
        return features

    def decode(self, features):
        coarse = self.decoder(features)
        coarse = torch.reshape(coarse, [-1, self.num_coarse, 3])
        fine = self.folding(features)
        return coarse, fine
