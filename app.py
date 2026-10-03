
import pandas as pd
import sqlite3
from companyClustering import CCBOST

# データ準備
with sqlite3.connect(r"./dataset/data.sqlite3") as conn:
    df = pd.read_sql('SELECT * FROM stockprice;', conn)
print(f"{df.shape= }")

# 学習＆予測
ccbost = CCBOST(df)
pred, name_list = ccbost.fit_and_predict()

# クラスタ要素表示
for i in range(pred.max() + 1): 
    print(f"Cluster{i} ==> {', '.join(name_list[pred == i])}")

# ２次元散布図
ccbost.draw_scatter_diagram()

# ３次元散布図
ccbost.draw_3D_scatter_graph()
