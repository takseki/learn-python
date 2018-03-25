# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer
from matplotlib.colors import ListedColormap

# テストデータ生成
class GenModel:
    # 2次元正規分布
    def generate(self, size):
        mean = np.array([-2., -2.])
        cov = np.diag([0.2, 0.2])
        g_size = size // 3
        y1 = np.random.multivariate_normal(mean, cov, g_size)
        mean = np.array([2., -1.5])
        y2 = np.random.multivariate_normal(mean, cov, g_size)
        mean = np.array([0, 1.5])
        y3 = np.random.multivariate_normal(mean, cov, size - g_size*2)
        return np.row_stack((y1, y2, y3))

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
            [[self.__kernel(X[i], X[j]) for j in range(N)] for i in range(N)])
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
            [[self.__kernel(x[i], self.X[j]) for j in range(N)] for i in range(Nx)]
        )                        # (Nx,N)
        # 中心化
        K = self.centerer.transform(K)
        # 主成分を計算
        return K.dot(self.a) / self.sigma  # (Nx,n)
    
if __name__ == '__main__':
    gen_model = GenModel()
    X = gen_model.generate(60)
    X = X - np.mean(X, axis=0)

    beta = 0.5
    pca = KernelPca(beta)
    y = pca.fit_transform(X, 2)
    y2 = pca.transform(X)

    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=beta)
    y1 = scikit_kpca.fit_transform(X)
    y_kpca = scikit_kpca.transform(X)

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


    # 主成分
    ax.axis([-1, 1, -1, 1])
    ax3.scatter(y[:,0], y[:, 1], marker=".")

    # PRML 12章にあるような元の空間での基底関数のプロットをしてみる
    resolution=0.1
    xx1, xx2 = np.meshgrid(np.arange(-4, 4, resolution),
                           np.arange(-4, 4, resolution))
    mesh = np.array([xx1.ravel(), xx2.ravel()]).T
    
    z = pca.transform(mesh)
    
    # デフォルトのcontour色だとデータ点と重なった時見づらい
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:1])
    
    # 上位成分を元空間上に等高線図でプロット
    for i in range(2):
        ax = axes[i]
        #ax.scatter(X[:, 0], X[:, 1], color='red', marker='^', alpha=0.5)
        z_i = z[:, i].reshape(xx1.shape)
        ax.pcolor(xx1, xx2, z_i, cmap='RdBu')

    ax1.scatter(X[:,0], X[:, 1], marker=".", color='black')
    ax2.scatter(X[:,0], X[:, 1], marker=".", color='black')

    plt.show()
    #plt.savefig('kernel_pca.png')
    
