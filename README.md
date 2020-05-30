# GLMとGLMMのゼロから実装
一般化線形モデル(GLM)と一般化線形混合モデル(GLMM)のゼロから実装して勉強してみる。

## 必要なPythonライブラリ
* pandas
* Statsmodels
* matplotlib
* scipy
* pypeR
  - R
  - glmmML

## 分析概要
「データ解析のための統計モデリング入門」より、植物のサイズ(x)より、
何割の種子数(y/N)が生存しているか予測を行う。
なおデータは環境の影響によりブロック差が生まれていると仮定されている。
そのため、GLMによる分散の予測結果は本来のデータの分散と大きくはずれている。
GLMMによるブロック差を考慮したモデリングを行うことによって、
データからでは観測されなかったブロック差を組み込むことができる。  

こちらよりデータを拝借いたしました。  
https://github.com/aviatesk/intro-statistical-modeling

## 結果
**GLMによる結果**  
どの個体も均質と仮定しているため、予測値が大きく外れている。
![GLM](https://github.com/Kageshimasu/glm-glmm-from-scratch/blob/master/images/glm.png)
**GLMMによる結果**  
どの個体も均質と仮定しているため、GLMMに比べて当てはまりが良い。
![GLMM](https://github.com/Kageshimasu/glm-glmm-from-scratch/blob/master/images/glmm.png)
