import joblib
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Meiryo'
import numpy as np
import os
import pandas as pd
import plotly.offline as po
import plotly.io as pio
import plotly.express as px
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


class CCBOST:
    def __init__(self, df, model_dir="./model"):
        # create mode save path
        self.model_path = os.path.join(model_dir, f"pipe_{sklearn.__version__}.pkl")
        
        # create scode name list
        self.name_list = df.drop_duplicates(subset=["銘柄コード"], keep="last")["銘柄名"].values
        
        # （終値 - 始値）差分データ列追加
        df["stock_price_close_open_diff"] = df["終値"] - df["始値"]
        
        # 銘柄コードごとに日付を特徴量とするデータフレーム作成
        self.df = df.set_index(['銘柄コード', '日付'], drop=True)['stock_price_close_open_diff'].unstack()


    def fit_and_predict(self):
        """define model, fit and predict"""
        # define model and fit
        self.pipe = Pipeline([
            ("preprocessing", StandardScaler()),
            ("estimator", KMeans(init="k-means++",  # converge quickly
                                 n_clusters=5,
                                 random_state=42,
                                 verbose=0))
            ]).fit(self.df)
        
        # print(f"{self.pipe= }")
        joblib.dump(self.pipe, self.model_path, compress=True)
        
        # predict
        self.pred = self.pipe.predict(self.df)
        print(f"{self.pred= }")
        
        return self.pred, self.name_list
    
    
    def draw_scatter_diagram(self, output_dir="./output"):
        """Draw scatter diagram with two features"""
        features = self.df.columns

        fig = plt.figure(figsize=(7, 5), facecolor="w")
        ax = fig.add_subplot(111)

        X_std = self.pipe["preprocessing"].transform(self.df)
        x = X_std [:, -1]  # N day
        y = X_std [:, -2]  # N-1 day

        ax.scatter(x, y, c=self.pipe["estimator"].labels_)

        # name
        for i, name in enumerate(self.name_list.tolist()):
            plt.text(x[i] + 0.1, y[i], name, fontsize=12)

        # centroids
        ax.scatter(self.pipe["estimator"].cluster_centers_[:, 0],
                   self.pipe["estimator"].cluster_centers_[:, 1],
                   s=250,
                   marker="*",
                   c="red",
                   edgecolor="black",
                   label="Centroids")

        ax.set_title("Scatter Diagram - KMeans")
        ax.set_xlabel(features[-1])
        ax.set_ylabel(features[-2])

        plt.legend(scatterpoints=1)
        plt.grid()
        plt.tight_layout()

        plt.savefig(Path(output_dir, "kmeans_scatter_diagram.png"))
        plt.close()
        
        
    def draw_3D_scatter_graph(self, output_dir=r"./output"):
        """Draw 3-D scatter graph with three features"""

        X_std = self.pipe["preprocessing"].transform(self.df)
        x = X_std[:, -1]  # N day
        y = X_std[:, -2]  # N-1 day
        z = X_std[:, -3]  # N-2 day
        c = self.pred

        fig = px.scatter_3d(self.df, x=x, y=y, z=z, color=c, hover_name=self.name_list.tolist(), title="Tips: the axes through SelectKBest algorithm.")
        po.plot(fig, filename=os.path.join(output_dir, "3DscatterDiagram.html"), auto_open=False)  # save as HTML
        pio.write_image(fig, os.path.join(output_dir, "3DscatterDiagram.png"))
