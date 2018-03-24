# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import operator as op

# テストデータ生成
class GenModel:
    # 2次元正規分布
    def generate(self, size):
        mean = np.zeros(2)
        c = np.cos(np.pi/6)
        s = np.sin(np.pi/6)
        R = np.array([[c, -s], [s,c]])
        # 回転させる
        cov = R.dot(np.diag([1.0, 0.2])).dot(R.T)
        y = np.random.multivariate_normal(mean, cov, size)
        return y

# 主成分分析
class Pca:
    # データを入力して主成分ベクトルを計算する
    # データは平均0を前提とする
    def fit(self, X):
        # 共分散行列
        Sigma = np.dot(X.T, X)
        vals, vecs = np.linalg.eigh(Sigma)
        # 固有値の大きい順にソート
        indices = vals.argsort()[::-1]
        vals = vals[indices]
        vecs = vecs[:, indices]
        # v[i] : 第i主成分の固有ベクトル
        self.v = np.array(vecs).T

    # x を主成分表示にして返す
    # n: 抽出する主成分の数
    def transform(self, x, n):
        return np.dot(x, self.v[:n].T)
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = gen_model.generate(500)
    x = x - np.mean(x, axis=0)

    pca = Pca()
    pca.fit(x)
    y = pca.transform(x, 2)
    
    fig, axes = plt.subplots(ncols=2, figsize=(8,4))

    for ax in axes:
        r = 4
        ax.axis([-r, r, -r, r])
        ax.grid(True)
        ax.set_aspect('equal')

    (ax1, ax2) = axes
    # 元データ
    ax1.scatter(x[:,0], x[:, 1], marker=".")
    # 固有ベクトルを描く
    num = pca.v.shape[0]
    ax1.quiver(np.zeros(num), np.zeros(num),
               4 * pca.v[:, 0], 4 * pca.v[:, 1], angles='xy', scale_units='xy', scale=1)

    # 主成分
    ax2.scatter(y[:,0], y[:, 1], marker=".")
    plt.show()
