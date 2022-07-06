from pyspark.sql import functions as F
from pyspark.sql.functions import rank,sum,col
from pyspark.sql import Window
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns


def plot_geomap(df, column, title):
    fig, ax = plt.subplots(figsize=(15, 10))
    vmin, vmax = df[column].min(), df[column].max()

    df.plot(column = column, linewidth=0.09, edgecolor='k', 
        figsize=(10, 10),         
        ax=ax, cmap='Blues', legend=True,
        legend_kwds={'shrink': 0.78})

    ax.set_title(title)

def plot_pivot_table_heatmap(df, v_col, h_col, value_col):
    df_aux = df.groupby([v_col, h_col], as_index=False)[value_col].mean()
    df_aux = df_aux.sort_values([v_col, h_col]).reset_index(drop=True)
    df_aux_heatmap = pd.pivot_table(df_aux, values=value_col, index=v_col, columns=h_col)

    if df_aux_heatmap.shape[0]>20:
        fig, ax = plt.subplots(figsize=(30, 24))
    else:
        fig, ax = plt.subplots(figsize=(12, 4))

    sns.heatmap(df_aux_heatmap, cbar_kws={'label': value_col}, 
        ax=ax, cmap='coolwarm')
    ax.set_ylabel(v_col)
    ax.set_xlabel(h_col)
    ax.set_title(f"{value_col} - {v_col} - {h_col}")
    plt.show()