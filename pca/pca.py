# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt

# テストデータ生成
class GenModel:
    # ノイズ無し
    def generate(self, size):
        mean = np.zeros(2)
        c = np.cos(np.pi/6)
        s = np.sin(np.pi/6)
        R = np.array([[c, -s], [s,c]])
        cov = R.dot(np.diag([10.0, 1.0])).dot(R.T)
        y = np.random.multivariate_normal(mean, cov, size)
        return y

# 主成分分析
class Pca:
    # データを入力して主成分ベクトルを計算する
    # データは平均0を前提とする
    def fit(self, X):
        # 共分散行列
        Sigma = np.dot(X.T, X)
        self.l, self.v = np.linalg.eigh(Sigma)
        indicies = np.argsort(self.l)
        self.l = [self.l[i] for i in indicies[::-1]]
        self.v = np.array([self.v[i] for i in indicies[::-1]])
        print('eigen values',  self.l)
        print('eigen vectors', self.v)

    # 主成分を返す
    def transform(self, x):
        return np.dot(x, self.v.T)
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = gen_model.generate(500)
    x = x - np.mean(x, axis=0)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
    
    ax1.scatter(x[:,0], x[:, 1], marker=".")
    ax1.axis([-10, 10, -10, 10])
    ax1.grid(True)
    
    pca = Pca()
    pca.fit(x)
    y = pca.transform(x)

    ax2.scatter(y[:,0], y[:, 1], marker=".")
    ax2.axis([-10, 10, -10, 10])
    ax2.grid(True)
    plt.show()
