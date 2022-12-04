import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


ITEMS_IN_BRANBD_THRESHOLD = 500
GRAPH__SETTINGS = {
    "edges_corr_threshold": 0.12
}


def data_read_clear(file_name: str) -> pd.DataFrame:
    brand_data_df = pd.read_csv(file_name)
    brand_data_df.head()
    # filter fields with empty order_id
    brand_data_df = (brand_data_df.loc[brand_data_df.order_id != 'none'].
                     reset_index())

    # reset item_cnt columns for 1 - was in order, 0 - wasn't
    brand_data_df = (brand_data_df.groupby(["order_id", "vendor"]).
                     agg(item_cnt=(
                         "item_id", lambda x: 1 if x.nunique() > 0 else 0
                         )).reset_index())

    # pivot by vendor names
    brand_data_df_pivot = (brand_data_df.pivot_table(values="item_cnt",
                                                    index=["order_id"], 
                                                    columns="vendor", 
                                                    fill_value=0).reset_index())

    # filter pivot table from non-popular brands
    cool_brands = brand_data_df_pivot.groupby(by='order_id').sum().sum()
    cool_brands = cool_brands.loc[cool_brands >= ITEMS_IN_BRANBD_THRESHOLD]
    brand_data_df_pivot_cool = (brand_data_df_pivot.
                                loc[:, brand_data_df_pivot.columns.
                                    isin(cool_brands.index)])

    return brand_data_df_pivot_cool.corr()


def plot_graph(corr_data) -> None:
    # create graph
    graph_full = nx.from_pandas_adjacency(corr_data, create_using=nx.DiGraph)
    graph_strong_edges = graph_full.copy()
    # remove weak connections
    graph_strong_edges.remove_edges_from(
        [(n1, n2) for n1, n2, w in graph_strong_edges.edges(data="weight") 
         if w < GRAPH__SETTINGS["edges_corr_threshold"]])
    # remove self loops on nodes
    graph_strong_edges.remove_edges_from(
        [(n1, n2) for n1, n2, w in 
         graph_strong_edges.edges(data="weight") if w == 1])
    # remove isolated nodes
    graph_strong_edges.remove_nodes_from(list(nx.isolates(graph_strong_edges)))
    
    # manual tuning to pretty display nodes and labels
    pos = nx.spring_layout(graph_strong_edges, 
                           scale=20, k=3/np.sqrt(graph_strong_edges.order()))
    nx.draw(graph_strong_edges,
            pos=pos, with_labels=True, node_color='lightgreen', node_size=100)
    plt.show()


if __name__ == "__main__":
    corr_ = data_read_clear("brand_data.csv")
    plot_graph(corr_)
