import pandas as pd 
import statsmodels.formula.api as smf
import matplotlib.pylab as plt
import numpy as np 
from scipy.stats import binom
import scipy.special as scm
from statistics import variance, mean
from scipy import integrate
from scipy.stats import norm
import scipy
from random import random
import pyper


def main():
    # 1. データのロード
    df = pd.read_csv('./data.csv')
    y = df['y'] / df['N']  # 比率にする
    x = df['x']
    N = df['N'][0]

    # 2. 使用する関数定義
    # 線形予測子を確率に変換するためのロジスティック関数
    logistic_func = lambda x: 1.0 / (1.0 + np.exp(-x + 1e-12))
    # 説明変数による確率の変化を考慮した二項分布
    binomial_func = lambda N, y, params, x, r: scm.comb(float(N), y) * logistic_func(params[0] + params[1] * x + r)**y * (1 - logistic_func(params[0] + params[1] * x + r))**(float(N) - y)
    # 畳み込みする関数 二項分布 x 正規分布
    integrated_func = lambda N, y, params, x, r: binomial_func(N, y, params, x, r) * norm.pdf(r, 0, params[2])
    # 尤度
    def calc_likelihood(params, N, y_vector, x_vector):
        likelihood = 1
        for i in range(len(y_vector)):
            likelihood *= integrate.quad(
                lambda r, N, y, x, params: integrated_func(N, y, params, x, r),
                -np.inf,
                np.inf,
                args=(N, y_vector[i], x_vector[i], params))[0]
        return likelihood

    # 3. pypeRを使って最尤推定をする
    r = pyper.R()
    r.assign("data", df)
    r("""
    library(glmmML)
    result <- glmmML(cbind(y, N - y) ~ x, data=data, family=binomial, cluster=id)
    """)
    result = r.get('result')
    a, b = result['coefficients']
    s = result['sigma']
    params = [a, b, s]
    print('推定されたパラメータ: {}'.format(params))
    print('最適化した尤度: {}'.format(calc_likelihood(params, N, df['y'], df['x'])))

    # 5. 正規分布のx軸について、-500~500分の確率分布の高さrを取得
    plot_num = 1000
    plot_x = np.linspace(0, 8, plot_num)
    r = [norm.pdf(i, 0, params[2]) for i in range(-plot_num // 2, plot_num // 2)]

    # 6. グラフプロットをする
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4), sharex=True)
    axL.plot(x, y, 'o', alpha=0.1)  # 説明変数と目的変数を散布図にプロットする 
    axL.plot(plot_x, logistic_func(params[0] + params[1] * plot_x + r))  # plot_num分関数をプロットする

    x_num = 4.0  # 説明変数を固定して、フィッテイングしているか見てみる(例えば4.0で試してみる)
    y_4 = df[df['x'] == x_num]['y'] 
    axR.hist(y_4) 
    x_list = np.array( # 固定された説明変数の時の確率分布を作る 0~8が横軸の範囲なので0~8の時の確率を計算する
        [integrate.quad(
            lambda r, N, y, x, params: integrated_func(N, y, params, x, r), -np.inf, np.inf, args=(N, _y, x_num, params))[0] for _y in range(0, N + 1)
            ]
        )
    axR.plot(range(0, N + 1), x_list * len(y_4))
    plt.show()

if __name__ == "__main__":
    main()