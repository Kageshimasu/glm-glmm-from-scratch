import pandas as pd 
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pylab as plt
import numpy as np 
import scipy.special as scm
from statistics import variance, mean

def main():
    df = pd.read_csv('./data.csv')
    y = df['y'] / df['N']  # 比率にする
    x = df['x']
    data = pd.concat([y, x], axis=1)

    # 1. 一般化線形モデル 最尤推定する
    glm_model = smf.glm('y ~ x', data=data, family=sm.families.Binomial())
    result = glm_model.fit(disp=0)
    print(result.params)
    print(result.summary())
    a, b = result.params
    logistic_func = lambda x: 1.0 / (1.0 + np.exp(-x))

    # 2. 葉数の数によって生存種子数がどう変化するかプロット
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4), sharex=True)
    axL.plot(x, y, 'o', alpha=0.1)
    plot_num = 1000
    plot_x = np.linspace(0, 8, plot_num)
    axL.plot(plot_x, logistic_func(a + b * plot_x))

    # 3. xを固定したときにyの生存種子数の分布の期待値と実際の数と照合
    x_num = 4.0  # 例えばxが4の個体数
    y_4 = df[df['x'] == x_num]['y']
    p_4 = logistic_func(a + b * x_num)
    N = df['N'][0]
    axR.hist(y_4) 
    x_list = np.array([scm.comb(float(N), _x) * p_4**_x * (1 - p_4)**(float(N) - _x) for _x in range(0, N + 1)])
    axR.plot(range(0, N + 1), x_list * len(y_4))

    # 4. 期待値と分散の比較
    print('確率分布が考える期待値: {}'.format(p_4 * N))
    print('実際のデータの期待値: {}'.format(mean(y_4)))
    print('確率分布が考える分散: {}'.format(p_4 * (1 - p_4) * N))
    print('実際のデータの分散: {}'.format(variance(y_4)))
    print('期待値はともかく、分散がでかすぎる')
    plt.show()

if __name__ == "__main__":
    main()