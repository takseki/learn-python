# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import operator as op
#import sklearn as sk
from sklearn.decomposition import KernelPCA

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

    # gauss kernel
    def __kernel(self, x1, x2):
        return np.exp(-self.beta * np.linalg.norm(x1 - x2)**2)

    # 中心化したグラム行列を返す
    def __gram_mat(self, X):
        N = X.shape[0]
        K = np.array(
            [[self.__kernel(X[i], X[j]) for i in range(N)] for j in range(N)])
        one = 1/N * np.ones(K.shape)
        return K - one.dot(K) - K.dot(one) + one.dot(K).dot(one)
        #K = np.fromfunction(lambda i, j:
        #                    self.__kernel(X[i], X[j]), (N, N), dtype=int)
    
    # データを入力して主成分ベクトルを計算する
    # n: 抽出する主成分の数
    def fit_transform(self, X, n):
        self.X = X
        # グラム行列
        K = self.__gram_mat(X)
        vals, vecs = np.linalg.eigh(K)
        # eighは固有値の昇順で出力される
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        # 特異値と左特異ベクトル
        self.sigma = np.sqrt(vals[:n]) # (n)
        self.a = np.array(vecs[:, :n]).T  # (n,N)
        return self.a.T * self.sigma   # (N,n)

    # xの主成分表示を返す
    def transform(self, x):
        N = self.X.shape[0]
        n = self.sigma.size
        normalized_a = self.a.T / self.sigma # (N,n)
        y = np.zeros(n)
        k = np.array([self.__kernel(x, self.X[i]) for i in range(N)]) #(N)
        y = k.dot(normalized_a) # (n)
        return y
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = gen_model.generate(100)
    x = x - np.mean(x, axis=0)

    beta=1.0
    pca = KernelPca(beta)
    y = pca.fit_transform(x, 2)

    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=beta)
    y_kpca = scikit_kpca.fit_transform(x)

    print(pca.transform(x[0]))
    print(scikit_kpca.transform([x[0]]))
    print(y[0])
    
    fig, axes = plt.subplots(ncols=3, figsize=(12,4))

    for ax in axes:
        r = 4
        ax.axis([-r, r, -r, r])
        ax.grid(True)
        ax.set_aspect('equal')

    (ax1, ax2, ax3) = axes
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
    
