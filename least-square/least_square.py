# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt

# テストデータ生成
class GenModel:
    # ノイズ無し
    def generate(self, x):
        a = 4.0
        b = 10.0
        y = a * x + b
        return y

    # ノイズつき
    def observe(self, x):
        y = self.generate(x) + np.random.normal(0, 1.0, x.shape)
        return y

# 最小2乗回帰
class LeastSquare:
    # データを線形基底関数による表示に直す
    def __data_vec(self, x):
        return np.array([1.0, x])

    # 計画行列 X を返す
    def __design_mat(self, xv):
        return np.array([self.__data_vec(x) for x in xv])

    # 学習データを入力して係数wを決定する
    def fit(self, x, y):
        X = self.__design_mat(x)
        
        # solve (X^t X) w = X^t y        
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    
    # 予測値を返す
    def predict(self, x):
        # 配列かどうかで分けている（ダサい）
        try:
            X = self.__data_vec(x)
        except TypeError:
            X = self.__data_vec(x)
        return np.dot(X, self.w)
    
if __name__ == '__main__':
    gen_model = GenModel()
    x = 0.1 * np.arange(100)
    y = gen_model.observe(x)
    # t = gen_model.generate(x)

    lls = LeastSquare()
    lls.fit(x, y)
    print('w', lls.w)
    
    y_pred = lls.predict(x)
    y_pred_ = lls.predict(x[10])  # サンプル単位で渡しても動くことの確認
    print('y_pred', y_pred_)
    
    plt.scatter(x, y, color='lightblue')
    plt.plot(x, y_pred, color='red')
    # plt.plot(x, t, color='red')
    
    plt.show()
    # plt.savefig('least_square.png')
