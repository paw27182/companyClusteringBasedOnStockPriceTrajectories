"""
class test
> cd ./companyClusteringBasedOnStockPriceTrajectories/tests
start example
> pytest test_companyClustering.py

> pytest -v -s test_companyClustering.py  # -v:"PASSED" -s: print()
> pytest test_companyClustering.py::TestCCBOST
> pytest test_companyClustering.py::TestCCBOST::test_fit  # class method test

Update: September 30, 2026
"""
import pandas as pd
import pytest
import sqlite3
from pathlib import Path

from ..companyClustering import CCBOST

@pytest.fixture()
def sample_data():
    """Create shared data and objects."""
    dbname = r"../dataset/data.sqlite3"
    with sqlite3.connect(dbname) as conn:
            df = pd.read_sql('SELECT * FROM stockprice;', conn)
    print(f"{df.shape= }")
    
    return {"dbname": dbname,
            "data": df,
            "model_dir": "../model",
            "output_dir": "../output"}

class TestCCBOST():
    def test_fit_and_predict(self, sample_data):
        # データ準備
        df = sample_data['data']
        model_path = sample_data['model_dir']
        output_dir = sample_data['output_dir']
        
        # 学習＆予測
        ccbost = CCBOST(df, model_path)
        pred, name_list = ccbost.fit_and_predict()
        
        # assert len(pred) == 8
        
        # クラスタ要素表示
        for i in range(pred.max() + 1): 
            print(f"Cluster{i} ==> {', '.join(name_list[pred == i])}")
        
        # ２次元散布図
        ccbost.draw_scatter_diagram(output_dir)
        
        # ３次元散布図
        ccbost.draw_3D_scatter_graph(output_dir)
