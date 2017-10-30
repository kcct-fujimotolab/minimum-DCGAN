from argparse import ArgumentParser
# 自作関数群
from utils.file import load_model
from utils.image import save_images, to_dirname
from utils.generation import generate
# 使用ネットワーク


def get_args():
    description = 'Generate images from DCGAN models'
    parser = ArgumentParser(description=description)
    parser.add_argument('-b', '--batch', type=int, default=64, help='number of generated images')
    parser.add_argument('-o', '--output', type=str, default='gen', help='output directory path')
    parser.add_argument('-a', '--acc', type=float, default=0, help='acceptable range of accuracy')
    return parser.parse_args()


def main():
    args = get_args()
    # パラメータ設定
    batch = args.batch # 出力枚数
    output_dirname = to_dirname(args.output) # 出力先ディレクトリ
    acc = args.acc # 許容出力精度
    # モデルを読み込み
    G = load_model('G_model.json')
    D = load_model('D_model.json')
    G.load_weights('G_weights.hdf5')
    D.load_weights('D_weights.hdf5')
    # 生成
    images = generate(G, batch=batch)
    save_images(images, name=output_dirname)


if __name__ == '__main__':
    main()