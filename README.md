# Directories

- nets
    - network definition files that can be called from train/test/predict scripts

# run with MNIST

## Train

- use simple FCN (784 -> 128 -> 10)
- use MNIST dataset
- train 10 epochs
- saving directory is created automatically.

```
python train.py fcn -d mnist -s [save dir] -ne 10
```

It will output train logs, checkpoints, model, summary and figures. Explore them.

## Tensorboard

```
tensorboard --logdir=/path/to/save/dir
```

## Predict

You can use trained models as image classifiers that take an image file as input and predict it.

- use trained FCN
- use sample data `[repo_root]/sample/image/mnist_9.jpg`
- save summary figure. saving directory must already exist.

```
python predict.py -t [repo_root]/sample/image/mnist_9.jpg -m [model path (h5)] -s [save dir]
```
