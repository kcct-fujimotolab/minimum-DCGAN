from argparse import ArgumentParser
from keras.optimizers import SGD, Adam
# 自作関数群
from utils.file import save_model
from utils.image import save_images, load_images, to_dirname
from utils.generation import generate
from utils.training import train
# 使用ネットワーク
from networks.relu import build_generator, build_discriminator, build_GAN


def get_args():
    description = 'Build DCGAN models and train'
    parser = ArgumentParser(description=description)
    parser.add_argument('-d', '--dim', type=int, default=100, help='generator input dimension')
    parser.add_argument('-z', '--size', type=int, nargs=2, default=[64, 64], help='image size during training')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=500, help='number of epochs')
    parser.add_argument('-s', '--save', type=int, default=20, help='snapshot taking interval')
    parser.add_argument('-i', '--input', type=str, default='images', help='data sets path')
    parser.add_argument('-o', '--output', type=str, default='gen', help='output directory path')
    return parser.parse_args()


def main():
    args = get_args()
    # パラメータ設定
    input_dim = args.dim # 入力ベクトルサイズ
    image_size = args.size # 画像サイズ
    batch = args.batch # 勾配更新までの回数
    epochs = args.epoch # データを周回する回数
    save_freq = args.save # スナップショットのタイミング
    input_dirname = to_dirname(args.input) # 読み込み先ディレクトリ
    output_dirname = to_dirname(args.output) # 出力先ディレクトリ
    # モデルを構築
    G = build_generator(input_dim=input_dim, output_size=image_size)
    D = build_discriminator(input_size=image_size)
    GAN = build_GAN(G, D)
    # モデルをコンパイル
    optimizer = Adam(lr=1e-5, beta_1=0.1)
    D.compile(loss='binary_crossentropy', optimizer=optimizer)
    # この値にするとうまくいく、正直職人芸
    optimizer = Adam(lr=1e-4, beta_1=0.5)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    # モデルを保存
    save_model(G, 'G_model.json')
    save_model(D, 'D_model.json')
    # データセットを読み込み
    images = load_images(name=input_dirname, size=image_size)
    # 学習開始
    for epoch in range(epochs):
        print('Epoch: '+str(epoch+1)+'/'+str(epochs))
        train(G, D, GAN, sets=images, batch=batch)
        if (epoch + 1) % save_freq == 0:
            # 一定間隔でスナップショットを撮る
            results = generate(G, batch=batch)
            save_images(results, name=output_dirname+str(epoch+1))
            G.save_weights('G_weights.hdf5')
            D.save_weights('D_weights.hdf5')


if __name__ == '__main__':
    main()