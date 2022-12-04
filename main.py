import pandas as pd
import numpy as np
import scipy.stats
from math import sqrt, factorial, isnan
from statsmodels.stats.proportion import proportion_confint
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from ipywidgets import Layout, widgets
from networkx.drawing.nx_agraph import graphviz_layout


brand_data_df = pd.read_csv("brand_data.csv")
brand_data_df.head()
brand_data_df = (brand_data_df.loc[brand_data_df.order_id != 'none'].
                 reset_index())

brand_data_df = (brand_data_df.groupby(["order_id", "vendor"]).
                 agg(item_cnt=("item_id", lambda x: 1 if x.nunique() > 0 else 0)).
                 reset_index())

brand_data_df_pivot = (brand_data_df.pivot_table(values="item_cnt",
                                                 index=["order_id"], 
                                                 columns="vendor", 
                                                 fill_value=0).reset_index())

brand_data_df_pivot.head().to_csv("brand_data_df_pivot_h.csv")
cool_brands = brand_data_df_pivot.groupby(by='order_id').sum().sum()
cool_brands = cool_brands.loc[cool_brands >= 500]

brand_data_df_pivot_cool = (brand_data_df_pivot.
                            loc[:, brand_data_df_pivot.columns.
                                isin(cool_brands.index)])
brand_data_df_pivot_cool.shape
corr_ = brand_data_df_pivot_cool.corr()
corr_.to_csv("corr.csv")
for idx_left in corr_.index:
    for idx_right in corr_.index:
        # if idx_left == idx_right:
        #     continue
        ones_left = cool_brands[idx_left]
        ones_right = cool_brands[idx_right]
        zeroes_right = brand_data_df_pivot_cool.shape[0] - ones_right
        common_ones = sum(brand_data_df_pivot_cool[idx_right] & brand_data_df_pivot_cool[idx_left])
        # ones_prob = (factorial(ones_right) / (factorial(ones_right - common_ones) * factorial(common_ones)) *
        #              factorial(zeroes_right) / (factorial(zeroes_right - (ones_left - common_ones)) * factorial(ones_left - common_ones)) / 
        #              (factorial(zeroes_right) / (factorial(zeroes_right - common_ones) * factorial(common_ones))))
        ones_prob = (factorial(ones_right) / (factorial(ones_right - common_ones) * factorial(common_ones)) *
                     factorial(zeroes_right - (ones_left - common_ones)) * factorial(ones_left - common_ones) *
                     factorial(zeroes_right - common_ones) * factorial(common_ones))
        # conf_int = 
# sns.heatmap(corr_)
# sns.clustermap(corr_, figsize=(20,20))
# plt.show()


G = nx.from_numpy_matrix(np.matrix(corr_), create_using=nx.DiGraph)
G = nx.from_pandas_adjacency(corr_, create_using=nx.DiGraph)
# layout = nx.spring_layout(G)
F = G.copy()
threshold = 0.12
F.remove_edges_from([(n1, n2) for n1, n2, w in F.edges(data="weight") if w < threshold])
F.remove_edges_from([(n1, n2) for n1, n2, w in F.edges(data="weight") if w == 1])
F.remove_nodes_from(list(nx.isolates(F)))


pos = graphviz_layout(F, prog="fdp")
label_ratio = 1.0/8.0
pos_labels = {} 
#For each node in the Graph
for aNode in F.nodes():
    #Get the node's position from the layout
    x,y = pos[aNode]
    #Get the node's neighbourhood
    N = F[aNode]
    #Find the centroid of the neighbourhood. The centroid is the average of the Neighbourhood's node's x and y coordinates respectively.
    #Please note: This could be optimised further
    cx = sum(map(lambda x:pos[x][0], N)) / len(pos)
    cy = sum(map(lambda x:pos[x][1], N)) / len(pos)
    #Get the centroid's 'direction' or 'slope'. That is, the direction TOWARDS the centroid FROM aNode.
    slopeY = (y-cy)
    slopeX = (x-cx)
    #Position the label at some distance along this line. Here, the label is positioned at about 1/8th of the distance.
    pos_labels[aNode] = (x+slopeX*label_ratio, y+slopeY*label_ratio)



# layout = nx.spring_layout(F)
# nx.draw(F, layout)
node_labels = dict(zip(F.nodes, list(F.nodes)))
# nx.draw_networkx_labels(F, layout, labels=node_labels, font_size=10, font_family='sans-serif')
# pos = nx.spring_layout(F, scale=20, k=3/np.sqrt(F.order()))
pos = nx.spring_layout(F, scale=20, k=3/np.sqrt(F.order()))
# nx.draw_networkx_labels(F, pos=pos_labels, font_size=2)
# nx.draw_networkx_labels(F, pos=pos, font_size=2)
# nx.draw_networkx(F, node_size=100)
nx.draw(F, pos=pos, with_labels=True, node_color='lightgreen', node_size=100)
plt.show()




proportion_confint(10, 120, alpha=0.005, method="wilson")
# 1/12 + scipy.stats.norm.ppf(0.975) * sqrt(1/12 * (1 - 1/12) / 100)
# 1/12 - scipy.stats.norm.ppf(0.975) * sqrt(1/12 * (1 - 1/12) / 100)

# corr = brand_data_df_pivot_cool.rolling(60).corr()
# print(corr)
# corr_ = np.array([corr.loc[i].to_numpy() for i in brand_data_df_pivot_cool.index if not np.isnan(corr.loc[i].to_numpy()).all()])
# corr_ = np.nansum(corr_, axis=0)/len(corr_)
# corr_ = pd.DataFrame(columns=brand_data_df_pivot_cool.columns.tolist(), index=brand_data_df_pivot_cool.columns.tolist(), data=corr_)
# print(corr_.shape, corr_.head())

