# DCGAN
Implementation of basic and universal DCGAN model using Keras.

## Description

These programs can load images from the specified directory, resize these images and train the model.
You can record images generated during training.
Also you can load trained models and generate images.

## Requirements

- Python 3.0 or more
- Keras 2.0 or more (Tensorflow backend)
- Pillow
- numpy
- tqdm
- h5py

## Get started

1. Clone this repository:
```sh
git clone https://github.com/kcct-fujimotolab/DCGAN.git
cd DCGAN/
```

2. Make a directory for data sets:
```sh
mkdir images
```

3. Collect images (more than thousands better):
```sh
ls images/
data0000.jpg   data0001.jpg   ...   data9999.jpg
```

4. Start training with specifying image size, number of epochs, data set directory, etc.:
```sh
python train.py --input images/ --size 64 64 --epoch 1000
```

5. Generate images with specifying output directory, number of batches:
```sh
python generate.py --output gen/ --batch 64
```

## Options

`--help` `-h`: show information

### train.py

`--input` `-i`: data sets path (default `-i images/`)  
`--size` `-z`: image size during training, **2 values required**, **must be multiples of 8** (default `-z 64 64`)  
`--epoch` `-e`: number of epochs (default `-e 500`)  
`--batch` `-b`: batch size (default `-b 64`)  
`--dim` `-d`: input dimension of generator (default `-d 100`)  
`--output` `-o`: output directory path (default `-i gen/`)  
`--save` `-s`: taking snapshot interval (default `-i 20`)

### generate.py

`--output` `-o`: output directory path (default `-o gen/`)  
`--batch` `-b`: number of generated images (default `-b 64`)

## Results

We extracted 4096 images from the face data provided by [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/), and trained.

### 30 epochs
![30](https://i.imgur.com/PrQkuVP.jpg)

### 600 epochs
![600](https://i.imgur.com/77kPHTO.jpg)

### 1200 epochs
![1200](https://i.imgur.com/TWri1m9.jpg)

### 1500 epochs
![1500](https://i.imgur.com/sKjWmpT.jpg)

## Author

[Fujimoto Lab](http://www.kobe-kosen.ac.jp/~fujimoto/) in [Kobe City College of Technology](http://www.kobe-kosen.ac.jp)  
Undergraduate Student of Electronics Department  
[@yoidea](https://twitter.com/yoidea)