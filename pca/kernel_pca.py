# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer

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

# カーネル主成分分析
class KernelPca:
    # beta: ガウスカーネルパラメータ
    def __init__(self, beta):
        self.beta = beta
        self.centerer = KernelCenterer()

    # gauss kernel
    def __kernel(self, x1, x2):
        return np.exp(-self.beta * np.linalg.norm(x1 - x2)**2)

    # データを入力して主成分ベクトルを計算する
    # shape(X) = (N, M)
    # n: 抽出する主成分の数
    def fit_transform(self, X, n):
        self.X = X
        # グラム行列
        N = X.shape[0]
        K = np.array(
            [[self.__kernel(X[i], X[j]) for i in range(N)] for j in range(N)])
        # 中心化
        K = self.centerer.fit_transform(K)
        # eighは固有値の昇順で出力される
        vals, vecs = np.linalg.eigh(K)
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        # 特異値と左特異ベクトル、上位n個
        self.sigma = np.sqrt(vals[:n])  # (n)
        self.a = np.array(vecs[:, :n])  # (N,n)
        return self.sigma * self.a      # (N,n)

    # xの主成分表示を返す
    # shape(x)=(Nx, M)
    def transform(self, x):
        # グラム行列
        N = self.X.shape[0]
        Nx = x.shape[0]
        K = np.array(
            [[self.__kernel(x[i], self.X[j]) for i in range(Nx)] for j in range(N)]
        )                        # (Nx,N)
        # 中心化
        K = self.centerer.transform(K)
        # 主成分を計算
        return K.dot(self.a) / self.sigma  # (Nx,n)
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = gen_model.generate(100)
    x = x - np.mean(x, axis=0)

    beta = 0.1
    pca = KernelPca(beta)
    y = pca.fit_transform(x, 2)
    y2 = pca.transform(x)

    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=beta)
    y1 = scikit_kpca.fit_transform(x)
    y_kpca = scikit_kpca.transform(x)

    print('        transform    ', y2[0])
    print('        fit_transform', y[0])
    print('sklearn transform    ', y_kpca[0])
    print('sklearn fit_transform', y1[0])

    fig, axes = plt.subplots(ncols=3, figsize=(12,4))

    for ax in axes:
        r = 4
        ax.axis([-r, r, -r, r])
        ax.grid(True)
        ax.set_aspect('equal')

    (ax1, ax2, ax3) = axes
    r = 1
    ax2.axis([-r, r, -r, r])
    ax3.axis([-r, r, -r, r])
    # 元データ
    ax1.scatter(x[:,0], x[:, 1], marker=".")
    # 固有ベクトルを描く
    # num = pca.v.shape[0]
    # ax1.quiver(np.zeros(num), np.zeros(num),
    # 4 * pca.v[:, 0], 4 * pca.v[:, 1], angles='xy', scale_units='xy', scale=1)

    # 主成分
    ax2.scatter(y[:,0], y[:, 1], marker=".")
    ax3.scatter(y_kpca[:,0], y_kpca[:, 1], marker=".")
    
    plt.show()
    # plt.savefig('kernel_pca.png')
    
