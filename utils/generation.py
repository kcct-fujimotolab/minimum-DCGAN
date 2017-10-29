import numpy as np


def generate(G, batch):
    """
    画像データを生成
    # 引数
        G : Keras model, 生成器
        batch : Integer, 出力画像枚数
    # 戻り値
        images : Numpy array, 画像データ
    """
    input_dim = G.input_shape[1]
    noise = np.random.uniform(0, 1, (batch, input_dim))
    images = G.predict(noise)
    images = images * 255
    return images