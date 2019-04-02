import argparse

def arg_parser():
    psr = argparse.ArgumentParser()
    psr.add_argument('net', help='name of network')
    psr.add_argument('-d', '--dataset', help='dataset directory', required=True)
    psr.add_argument('-s', '--save_dir', help='directory where the models/checkpoints are saved', required=True)
    psr.add_argument('-bs', '--batch_size', help='batch size', type=int, default=32)
    psr.add_argument('-ne', '--num_epochs', help='the number of training epochs', type=int, default=1)
    psr.add_argument('-ts', '--target_size', help='target image size. This argument needs to be set only if `dataset` is a directory path.', nargs='+', type=int, default=[224, 224])
    psr.add_argument('-cm', '--class_mode', help='classification mode. This argument needs to be set only if `dataset` is a directory path. See "data_generator.py" for more details.', default='categorical')
    psr.add_argument('-td', '--test', help='test dataset directory', required=False, default='')
    psr.add_argument('-vd', '--validation', help='validation dataset directory', required=False, default='')
    psr.add_argument('-wt', '--weights', help='pre-trained weights for fine-tuning. supported only "imagenet".', required=False, default='')
    psr.add_argument('-fc', '--full_conn', help='specify last full connection layers structure except output layer. This argument will be used only when `weights` is "imagenet"', nargs='+', type=float, default=[2048])

    return psr

def arg_parser_pred():
    psr = argparse.ArgumentParser()
    psr.add_argument('-t', '--target', help='target file to input', required=True)
    psr.add_argument('-m', '--model_path', help='model path to use for prediction', required=True)
    psr.add_argument('-s', '--save_dir', help='directory to save prediction result', default='')
    psr.add_argument('-is', '--img_size', help='imgage width and height', nargs='+', type=int, default=[28, 28])
    psr.add_argument('-cm', '--color_mode', help='color mode. one of "grayscale", "rgb", "rgba".', default='grayscale')
    psr.add_argument('-nc', '--num_classes', help='no. of classes', type=int, default=10)

    return psr

