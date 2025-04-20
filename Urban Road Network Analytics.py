#!/usr/bin/env python
# coding: utf-8

# In[ ]:




get_ipython().system('pip install osmnx geopandas pandas seaborn matplotlib plotly streamlit shapely scikit-learn networkx contextily')


# In[ ]:


import osmnx as ox
import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from shapely import wkt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import networkx as nx
import contextily as ctx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


# In[ ]:


def perform_geospatial_analysis(gdf_edges, gdf_nodes):
    # Sample data to reduce size
    gdf_sample = gdf_edges.sample(frac=0.5, random_state=42)

    # Plot Road Network (Interactive with Plotly)
    fig = px.scatter_mapbox(
        gdf_sample,
        lat=gdf_sample.geometry.centroid.y,
        lon=gdf_sample.geometry.centroid.x,
        color="traffic_density",
        size="traffic_density",
        hover_data=["maxspeed", "lanes", "congestion_level"],
        mapbox_style="open-street-map",
        zoom=10,
        title="Road Network Map (Interactive)",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.show()

    # Calculate Centrality Measures
    graph = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
    nodes_sample = list(graph.nodes())[:500]  # Sample nodes for faster processing
    centrality = nx.betweenness_centrality_subset(graph, sources=nodes_sample, targets=nodes_sample, weight="length")

    # 🔹 Fix: Ensure 'u' and 'v' columns exist before mapping centrality
    if "u" in gdf_edges.columns and "v" in gdf_edges.columns:
        gdf_edges["centrality"] = gdf_edges["u"].map(centrality).fillna(0)
    else:
        print("Warning: 'u' and 'v' columns not found in gdf_edges. Skipping centrality calculation.")
        gdf_edges["centrality"] = 0  # Default centrality to zero

    # Plot Centrality (Interactive with Plotly)
    fig = px.scatter_mapbox(
        gdf_edges,
        lat=gdf_edges.geometry.centroid.y,
        lon=gdf_edges.geometry.centroid.x,
        color="centrality",
        size="centrality",
        hover_data=["maxspeed", "lanes", "congestion_level"],
        mapbox_style="open-street-map",
        zoom=10,
        title="Road Centrality Map (Interactive)",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.show()

    # Additional Insights: Centrality Distribution
    fig = px.histogram(
        gdf_edges,
        x="centrality",
        nbins=50,
        title="Centrality Distribution",
        labels={"centrality": "Betweenness Centrality"},
    )
    fig.show()

    # Additional Insights: Traffic Density vs. Centrality
    fig = px.scatter(
        gdf_edges,
        x="traffic_density",
        y="centrality",
        color="congestion_level",
        title="Traffic Density vs. Centrality",
        labels={"traffic_density": "Traffic Density", "centrality": "Betweenness Centrality"},
    )
    fig.show()

    # Additional Insights: Congestion Level Distribution
    congestion_counts = gdf_edges["congestion_level"].value_counts()
    fig = px.pie(
        congestion_counts,
        values=congestion_counts.values,
        names=congestion_counts.index,
        title="Congestion Level Distribution",
    )
    fig.show()

# Run analysis
perform_geospatial_analysis(gdf_edges, gdf_nodes)


# In[ ]:




