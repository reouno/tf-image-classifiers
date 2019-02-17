import argparse

def arg_parser():
    psr = argparse.ArgumentParser()
    psr.add_argument('net', help='name of network')
    psr.add_argument('-d', '--dataset', help='dataset directory', required=True)
    psr.add_argument('-s', '--save_dir', help='directory where the models/checkpoints are saved', required=True)
    psr.add_argument('-bs', '--batch_size', help='batch size', type=int, default=32)
    psr.add_argument('-ne', '--num_epochs', help='the number of training epochs', type=int, default=1)

    return psr

def arg_parser_pred():
    psr = argparse.ArgumentParser()
    psr.add_argument('-t', '--target', help='target file to input', required=True)
    psr.add_argument('-m', '--model_path', help='model path to use for prediction', required=True)
    psr.add_argument('-s', '--save_dir', help='directory to save prediction result', default='')
    psr.add_argument('-ih', '--img_height', help='input image height for the network', default=28, type=int)
    psr.add_argument('-iw', '--img_width', help='input image width for the network', default=28, type=int)

    return psr

