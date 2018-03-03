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
    # beta
    def __init__(self, beta, lam):
        self.beta = beta
        self.lam = lam
        
    # データを多項式基底関数による表示に直す
    def __kernel(self, x1, x2):
        return np.exp(-self.beta * (x1 - x2)**2)

    # グラム行列 K を返す
    def __gram_mat(self, x):
        N = x.size
        K = np.empty((N, N))
        for i in np.arange(N):
            for j in np.arange(N):
                K[i, j] = self.__kernel(x[i], x[j])
        return K

    # 学習データを入力して係数wを決定する
    def fit(self, x, y):
        self.N = x.size
        self.x = x
        K = self.__gram_mat(x)
        
        # solve (K + lambda I) a = y        
        self.a = np.linalg.solve(K + self.lam * np.eye(self.N), y)
        
    # 予測値を返す
    def predict(self, x):
        y = np.empty(x.shape)
        for i in np.arange(x.size):
            y[i] = 0
            for j in np.arange(self.N):
                y[i] += self.__kernel(x[i], self.x[j]) * self.a[j]
        return y
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = 0.1 * np.arange(100)
    y = gen_model.observe(x)
    t = gen_model.generate(x)

    kr = KernelRegression(1.0, 0.1)
    kr.fit(x, y)
    print('a', kr.a)
    
    y_pred = kr.predict(x)
    plt.scatter(x, y, color='lightblue')    
    plt.plot(x, y_pred, color='red')
    plt.plot(x, t, color='blue')
    
    plt.show()
    # plt.savefig('kernel_regression_poly.png')
