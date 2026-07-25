import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("🛒 Association Rule Mining Dashboard")

# Initialize Spark Session
@st.cache_resource
def get_spark_session():
    spark = SparkSession.builder \
        .appName("StreamlitFPGrowth") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

spark = get_spark_session()

# --- Sidebar for Parameters ---
st.sidebar.header("Adjust Parameters")
min_support_input = st.sidebar.slider("Minimum Support", min_value=0.01, max_value=0.1, value=0.02, step=0.005)
min_confidence_input = st.sidebar.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.3, step=0.05)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    raw_df = pd.read_csv('groceries - groceries (1).csv')

    transactions = []
    for index, row in raw_df.iterrows():
        num_items = int(row['Item(s)'])
        items_in_row = []
        for i in range(1, num_items + 1):
            item_col_name = f'Item {i}'
            item = row[item_col_name]
            if pd.notna(item):
                items_in_row.append(item)
        if items_in_row:
            transactions.append(items_in_row)

    pd_transactions_df = pd.DataFrame({'items': transactions})
    dataset = spark.createDataFrame(pd_transactions_df)

    cleaned_baskets = dataset.withColumn("items",
                                         F.array_distinct(
                                             F.transform(
                                                 F.col("items"),
                                                 lambda x: F.lower(F.trim(x))
                                             )
                                         ))
    final_baskets = cleaned_baskets.filter(F.size("items") > 1)
    return final_baskets

final_baskets_df = load_and_preprocess_data()

# --- FPGrowth Model ---
@st.cache_data(show_spinner="Generating Frequent Itemsets and Rules...")
def run_fpgrowth(data, support, confidence):
    fpGrowth = FPGrowth(itemsCol="items", minSupport=support, minConfidence=confidence)
    model = fpGrowth.fit(data)
    
    # Convert Spark DataFrames to Pandas for easier plotting and display in Streamlit
    frequent_itemsets_pd = model.freqItemsets.toPandas()
    association_rules_pd = model.associationRules.toPandas()
    
    return frequent_itemsets_pd, association_rules_pd, model

frequent_itemsets_pd, association_rules_pd, fp_model = run_fpgrowth(final_baskets_df, min_support_input, min_confidence_input)

st.subheader(f"Frequent Itemsets (Min Support: {min_support_input})")
st.write(f"Found {len(frequent_itemsets_pd)} frequent itemsets.")
st.dataframe(frequent_itemsets_pd.head(10))

st.subheader(f"Association Rules (Min Confidence: {min_confidence_input})")
st.write(f"Found {len(association_rules_pd)} association rules.")
st.dataframe(association_rules_pd.head(10))

# --- Visualizations ---
st.header("Visualizations")

# 1. Top 20 Most Frequent Items
st.subheader("Top 20 Most Frequent Items")
# Explode the items column to count individual item frequencies
exploded_items = final_baskets_df.withColumn("item", F.explode("items"))
item_counts = exploded_items.groupBy("item").count().orderBy(F.desc("count"))
top_items_pd = item_counts.limit(20).toPandas()

fig_top_items, ax_top_items = plt.subplots(figsize=(12, 7))
sns.barplot(x='count', y='item', data=top_items_pd, palette='viridis', ax=ax_top_items)
ax_top_items.set_title('Top 20 Most Frequent Items')
ax_top_items.set_xlabel('Frequency')
ax_top_items.set_ylabel('Item')
st.pyplot(fig_top_items)

# 2. Top Association Rules by Lift
st.subheader("Top Association Rules by Lift")
if not association_rules_pd.empty:
    top_rules_by_lift_pd = association_rules_pd.orderBy('lift', ascending=False).head(10)
    top_rules_by_lift_pd['rule_str'] = top_rules_by_lift_pd.apply(lambda row: f"{', '.join(row['antecedent'])} => {', '.join(row['consequent'])}", axis=1)

    fig_lift, ax_lift = plt.subplots(figsize=(12, 8))
    sns.barplot(x='lift', y='rule_str', data=top_rules_by_lift_pd, palette='magma', ax=ax_lift)
    ax_lift.set_title('Top Association Rules by Lift')
    ax_lift.set_xlabel('Lift')
    ax_lift.set_ylabel('Association Rule')
    st.pyplot(fig_lift)
else:
    st.write("No association rules found to plot.")

# 3. Support vs Confidence Scatter Plot
st.subheader("Support vs Confidence for Association Rules")
if not association_rules_pd.empty:
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='support', y='confidence', hue='lift', data=association_rules_pd, size='lift', sizes=(50, 400), palette='viridis', legend='full', ax=ax_scatter)
    ax_scatter.set_title('Support vs Confidence for Association Rules (Colored by Lift)')
    ax_scatter.set_xlabel('Support')
    ax_scatter.set_ylabel('Confidence')
    ax_scatter.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_scatter)
else:
    st.write("No association rules found to plot.")

# 4. Interactive Association Rule Network Graph (Plotly)
st.subheader("Interactive Association Rule Network Graph")
if not association_rules_pd.empty:
    G = nx.DiGraph()

    for index, row in association_rules_pd.iterrows():
        antecedent = frozenset(row['antecedent'])
        consequent = frozenset(row['consequent'])

        for item in antecedent:
            if item not in G:
                G.add_node(item, type='item')
        for item in consequent:
            if item not in G:
                G.add_node(item, type='item')

        antecedent_str = ', '.join(sorted(list(antecedent)))
        consequent_str = ', '.join(sorted(list(consequent)))

        if antecedent_str not in G:
            G.add_node(antecedent_str, type='itemset')
        if consequent_str not in G:
            G.add_node(consequent_str, type='itemset')

        G.add_edge(antecedent_str, consequent_str, weight=row['lift'], confidence=row['confidence'])

    pos = nx.spring_layout(G, k=0.5, iterations=100, scale=2)

    # Create a Plotly figure
    plotly_fig = go.Figure()

    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_colors.append('lightblue' if G.nodes[node].get('type') == 'item' else 'lightgreen')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_colors,
            size=30,
            line_width=2),
        hoverinfo='text'
    )

    plotly_fig.add_trace(node_trace)

    # Add edges - EACH EDGE AS A SEPARATE TRACE for individual coloring
    plotly_cmap = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']

    all_confidence_values = [d['confidence'] for u, v, d in G.edges(data=True)]
    min_conf = min(all_confidence_values) if all_confidence_values else 0
    max_conf = max(all_confidence_values) if all_confidence_values else 1

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        current_confidence = d['confidence']
        if max_conf - min_conf > 0:
            normalized_conf = (current_confidence - min_conf) / (max_conf - min_conf)
        else:
            normalized_conf = 0.5

        color_index = int(normalized_conf * (len(plotly_cmap) - 1))
        edge_color = plotly_cmap[color_index]

        hover_text = f"{u} -> {v}<br>Lift: {d['weight']:.2f}<br>Confidence: {d['confidence']:.2f}"

        plotly_fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=d['weight'] * 2.5, color=edge_color),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))
    
    # Add a color bar for confidence
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=plotly_cmap,
            showscale=True,
            cmin=min_conf,
            cmax=max_conf,
            colorbar=dict(
                title="Confidence",
                thickness=15,
                len=0.5,
                x=1.02,
                xanchor="left"
            )
        ),
        hoverinfo='none',
        showlegend=False
    )
    plotly_fig.add_trace(colorbar_trace)

    plotly_fig.update_layout(
        title='Interactive Association Rule Network Graph (Edge width by Lift, Color by Confidence)',
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    st.plotly_chart(plotly_fig, use_container_width=True)
else:
    st.write("No association rules found to generate network graph.")
