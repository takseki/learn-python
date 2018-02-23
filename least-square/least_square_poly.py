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

# 多項式最小2乗回帰
class LeastSquare:
    # N: 多項式次数
    def __init__(self, N):
        self.N = N
        
    # データを多項式基底関数による表示に直す
    def __data_vec(self, x):
        return np.array([x**n for n in range(self.N + 1)])

    # 計画行列 X を返す
    def __design_mat(self, xv):
        return np.array([self.__data_vec(x) for x in xv])

    # 学習データを入力して係数wを決定する
    def fit(self, x, y):
        X = self.__design_mat(x)
        print(X.shape)
        
        # solve (X^t X) w = X^t y        
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        
        # X^t (X X^t)^-1 y
        #self.w = np.dot(X.T, np.linalg.solve(np.dot(X, X.T), y))
        
    # 予測値を返す
    def predict(self, x):
        # 配列かどうかで分けている（ダサい）
        try:
            X = self.__design_mat(x)
        except TypeError:
            X = self.__data_vec(x)
        return np.dot(X, self.w)
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = 0.1 * np.arange(100)
    y = gen_model.observe(x)
    t = gen_model.generate(x)

    lls = LeastSquare(5)
    lls.fit(x, y)
    print('w', lls.w)
    
    y_pred = lls.predict(x)
    plt.scatter(x, y, color='lightblue')    
    plt.plot(x, y_pred, color='red')
    plt.plot(x, t, color='blue')
    
    plt.show()
    # plt.savefig('least_square_poly.png')
