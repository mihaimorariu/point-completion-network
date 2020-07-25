import argparse

import time

from common.helper import *


def parse_arguments():
    model_types = ['pcn_cd', 'pcn_emd', 'folding', 'fc']

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device',
                        type=int, default=0, help='device id (-1 for cpu, non-zero value for gpu)')
    parser.add_argument('-i', '--input-file',
                        type=str, default='demo_data/chair.pcd', help='input (partial) point cloud file')
    parser.add_argument('-m', '--model-type',
                        type=str, default='pcn_emd', choices=model_types, help='model type')
    parser.add_argument('-c', '--checkpoint',
                        type=str, default='data/trained_models/pcn_emd', help='checkpoint file')
    parser.add_argument('-p', '--num-gt-points',
                        type=int, default=16384, help='number of ground-truth points')
    parser.add_argument('-o', '--onnx-model-file',
                        type=str, default='model_tf.onnx', help='weights file for the exported TF->ONNX model')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()

    if args.device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.device))

    if args.model_type == 'pcn_cd':
        from pytorch_models.pcn_cd import Model
    elif args.model_type == 'pcn_emd':
        from pytorch_models.pcn_emd import Model

    print('Model type:', args.model_type)

    model = Model()
    if args.device >= 0:
        model.to(device)

    if args.onnx_model_file:
        model = convert_model_onnx_to_torch(model, args.onnx_model_file)

    partial = read_cloud_points(args.input_file)
    inputs = torch.Tensor(partial)
    inputs = inputs.transpose(1, 0)[None, :]
    inputs = inputs.to(device)

    time_beg = time.time()
    coarse, fine = model(inputs)
    time_end = time.time()
    print('Inference took {} seconds.'.format(time_end - time_beg))

    fine = fine.detach().cpu().numpy()[0]
    # coarse = coarse.detach().cpu().numpy()[0]

    plot_side_by_side(partial, fine)
