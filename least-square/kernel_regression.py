# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt

# テストデータ生成
class GenModel:
    # ノイズ無し
    def generate(self, x):
        a = 4.0
        b = 10.0
        y = a * np.sin(x) + b
        return y

    # ノイズつき
    def observe(self, x):
        y = self.generate(x) + np.random.normal(0, 1.0, x.shape)
        return y

# ガウスカーネル回帰
class KernelRegression:
    # beta: ガウスカーネルパラメータ
    # lam : 正則化係数
    def __init__(self, beta, lam):
        self.beta = beta
        self.lam = lam
        
    # gauss kernel
    def __kernel(self, x1, x2):
        return np.exp(-self.beta * (x1 - x2)**2)

    # グラム行列を返す
    def __gram_mat(self, x):
        N = x.size
        K = np.fromfunction(lambda i, j: self.__kernel(x[i], x[j]), (N, N), dtype=int)
        return K

    # 学習データを入力して係数aを決定する
    def fit(self, x, y):
        # 学習データを保持する
        self.x = x

        # solve (K + lambda I) a = y
        K = self.__gram_mat(x)
        self.a = np.linalg.solve(K + self.lam * np.eye(x.size), y)
        
    # 予測値を返す
    def predict(self, x):
        # \sum_j k(x, k_j) a_j
        K = np.fromfunction(lambda i, j: self.__kernel(x[i], self.x[j]),
                            (x.size, self.x.size), dtype=int)
        return K.dot(self.a)
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = 0.1 * np.arange(100)
    y = gen_model.observe(x)
    t = gen_model.generate(x)

    kr = KernelRegression(beta=1.0, lam=0.1)
    kr.fit(x, y)
    print('a', kr.a)
    
    y_pred = kr.predict(x)
    plt.scatter(x, y, color='lightblue')    
    plt.plot(x, y_pred, color='red')
    plt.plot(x, t, color='blue')
    
    plt.show()
    # plt.savefig('kernel_regression.png')
