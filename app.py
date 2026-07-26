import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import io

# Page Configuration
st.set_page_config(
    page_title="BasketSense - Association Rule Mining",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 50%, #00D2FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0 5px 0;
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 20px;
        font-style: italic;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 15px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #6B73FF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 5px;
    }
    .metric-delta {
        font-size: 0.85rem;
        color: #28a745;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 3px solid #6B73FF;
        display: inline-block;
    }
    
    .subsection-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3436;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #6B73FF;
        color: white;
    }
    
    /* Custom button */
    .download-btn {
        background: linear-gradient(135deg, #6B73FF, #00D2FF);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .download-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(107, 115, 255, 0.4);
    }
    
    /* Info box */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #00D2FF;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🛒 BasketSense</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Discovering Hidden Shopping Patterns with Association Rule Mining</div>', unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SPARK SESSION
# ============================================================================

@st.cache_resource
def get_spark_session():
    spark = SparkSession.builder \
        .appName("BasketSense") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

spark = get_spark_session()

# ============================================================================
# SIDEBAR - Parameters
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Parameters")
    st.markdown("---")
    
    min_support_input = st.slider(
        "Minimum Support", 
        min_value=0.005, 
        max_value=0.1, 
        value=0.02, 
        step=0.005,
        help="Lower values find more frequent itemsets"
    )
    
    min_confidence_input = st.slider(
        "Minimum Confidence", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Higher values give more reliable rules"
    )
    
    st.markdown("---")
    
    # Parameter Optimization - Removed button, just showing message
    st.markdown("### 📊 Parameter Tips")
    st.info("💡 Try different support and confidence ranges to discover more or fewer rules. Lower support finds more itemsets, higher confidence gives more reliable rules.")
    
    st.markdown("---")
    
    # Download Section
    st.markdown("### 📥 Export Results")
    
    if 'association_rules_pd' in globals() and not association_rules_pd.empty:
        csv_rules = association_rules_pd.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Rules (CSV)",
            data=csv_rules,
            file_name='basketsense_rules.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    if 'frequent_itemsets_pd' in globals() and not frequent_itemsets_pd.empty:
        csv_freq = frequent_itemsets_pd.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📋 Download Itemsets (CSV)",
            data=csv_freq,
            file_name='basketsense_itemsets.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    st.markdown("---")
    st.caption("Built with PySpark & Streamlit")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

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
    
    return final_baskets.toPandas()

final_baskets_df_pd = load_and_preprocess_data()
final_baskets_df = spark.createDataFrame(final_baskets_df_pd)

# ============================================================================
# FPGROWTH MODEL
# ============================================================================

@st.cache_data(show_spinner="Mining frequent itemsets and generating rules...")
def run_fpgrowth(data_pd, support, confidence):
    data = spark.createDataFrame(data_pd)
    fpGrowth = FPGrowth(itemsCol="items", minSupport=support, minConfidence=confidence)
    model = fpGrowth.fit(data)
    
    frequent_itemsets_pd = model.freqItemsets.toPandas()
    association_rules_pd = model.associationRules.toPandas()
    
    return frequent_itemsets_pd, association_rules_pd

frequent_itemsets_pd, association_rules_pd = run_fpgrowth(
    final_baskets_df_pd, 
    min_support_input, 
    min_confidence_input
)

# ============================================================================
# SUMMARY METRICS
# ============================================================================

st.markdown("### 📊 Dashboard Overview")

# Calculate metrics
total_transactions = len(final_baskets_df_pd)
all_items = set().union(*final_baskets_df_pd['items'].tolist())
unique_items = len(all_items)
avg_basket_size = final_baskets_df_pd['items'].str.len().mean()
total_rules = len(association_rules_pd)
total_itemsets = len(frequent_itemsets_pd)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📦 Transactions</div>
        <div class="metric-value">{total_transactions:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏷️ Unique Items</div>
        <div class="metric-value">{unique_items:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 Avg Basket Size</div>
        <div class="metric-value">{avg_basket_size:.1f}</div>
        <div class="metric-delta">items per transaction</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #00D2FF;">
        <div class="metric-label">📈 Itemsets Found</div>
        <div class="metric-value">{total_itemsets:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #FF6B6B;">
        <div class="metric-label">🔗 Rules Found</div>
        <div class="metric-value">{total_rules:,}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TABS FOR ORGANIZED CONTENT
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Itemsets & Rules",
    "📊 Visualizations",
    "🔍 Rule Explorer",
    "🌐 Network Graph",
    "📈 Advanced Analytics"
])

# ============================================================================
# TAB 1: ITEMSETS & RULES
# ============================================================================

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Frequent Itemsets")
        st.caption(f"Showing top 20 of {len(frequent_itemsets_pd)} itemsets")
        
        # Sort and display
        display_itemsets = frequent_itemsets_pd.nlargest(20, 'freq')
        display_itemsets['items_str'] = display_itemsets['items'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        st.dataframe(
            display_itemsets[['items_str', 'freq']].rename(
                columns={'items_str': 'Items', 'freq': 'Frequency'}
            ),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("### 📈 Association Rules")
        st.caption(f"Showing top 20 of {len(association_rules_pd)} rules")
        
        if not association_rules_pd.empty:
            display_rules = association_rules_pd.nlargest(20, 'lift')
            display_rules['antecedent_str'] = display_rules['antecedent'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
            display_rules['consequent_str'] = display_rules['consequent'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
            st.dataframe(
                display_rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].rename(
                    columns={
                        'antecedent_str': 'Antecedent',
                        'consequent_str': 'Consequent',
                        'support': 'Support',
                        'confidence': 'Confidence',
                        'lift': 'Lift'
                    }
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No association rules found with current parameters. Try lowering the support or confidence.")

# ============================================================================
# TAB 2: VISUALIZATIONS
# ============================================================================

with tab2:
    # Top Items
    st.markdown("### 🏆 Top 20 Most Frequent Items")
    
    exploded_items = final_baskets_df.withColumn("item", F.explode("items"))
    item_counts = exploded_items.groupBy("item").count().orderBy(F.desc("count"))
    top_items_pd = item_counts.limit(20).toPandas()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_top_items, ax_top_items = plt.subplots(figsize=(10, 7))
        colors = sns.color_palette("viridis", n_colors=len(top_items_pd))
        bars = ax_top_items.barh(top_items_pd['item'], top_items_pd['count'], color=colors)
        ax_top_items.set_title('Top 20 Most Frequent Items', fontsize=16, fontweight='bold')
        ax_top_items.set_xlabel('Frequency', fontsize=12)
        ax_top_items.set_ylabel('Item', fontsize=12)
        ax_top_items.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax_top_items.text(width + 1, bar.get_y() + bar.get_height()/2, 
                            f'{int(width)}', ha='left', va='center', fontsize=10)
        
        st.pyplot(fig_top_items)
        plt.close()
    
    with col2:
        # Quick stats about items
        st.markdown("### 📊 Item Statistics")
        total_item_occurrences = item_counts.agg(F.sum("count")).collect()[0][0]
        avg_occurrence = total_item_occurrences / unique_items
        
        st.metric("Total Item Occurrences", f"{total_item_occurrences:,}")
        st.metric("Avg Occurrences per Item", f"{avg_occurrence:.1f}")
        st.metric("Most Frequent Item", f"{top_items_pd.iloc[0]['item']} ({top_items_pd.iloc[0]['count']}x)")
        st.metric("Least Frequent (Top 20)", f"{top_items_pd.iloc[-1]['item']} ({top_items_pd.iloc[-1]['count']}x)")
    
    # Top 10 Item Share - Pie Chart
    st.markdown("### 📊 Top 10 Item Share")
    
    top_10_items = top_items_pd.head(10).copy()
    other_count = top_items_pd['count'].iloc[10:].sum() if len(top_items_pd) > 10 else 0
    
    if other_count > 0:
        top_10_items = pd.concat([
            top_10_items,
            pd.DataFrame({'item': ['Others'], 'count': [other_count]})
        ], ignore_index=True)
    
    fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
    colors_pie = sns.color_palette("viridis", n_colors=len(top_10_items))
    wedges, texts, autotexts = ax_pie.pie(
        top_10_items['count'],
        labels=top_10_items['item'],
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        textprops={'fontsize': 10}
    )
    ax_pie.set_title('Top 10 Items Share Distribution', fontsize=16, fontweight='bold')
    st.pyplot(fig_pie)
    plt.close()
    
    # Basket Size Distribution
    st.markdown("### 📦 Basket Size Distribution")
    
    basket_sizes = final_baskets_df.withColumn("basket_size", F.size("items")) \
                                  .groupBy("basket_size").count() \
                                  .orderBy("basket_size").toPandas()
    
    fig_sizes, ax_sizes = plt.subplots(figsize=(12, 6))
    bars = ax_sizes.bar(basket_sizes['basket_size'], basket_sizes['count'], 
                        color=sns.color_palette("Blues", n_colors=len(basket_sizes)))
    ax_sizes.set_title('Distribution of Basket Sizes', fontsize=16, fontweight='bold')
    ax_sizes.set_xlabel('Number of Items in Basket', fontsize=12)
    ax_sizes.set_ylabel('Number of Transactions', fontsize=12)
    ax_sizes.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax_sizes.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig_sizes)
    plt.close()
    
    # Support vs Confidence
    st.markdown("### 📈 Support vs Confidence (Colored by Lift)")
    
    if not association_rules_pd.empty:
        fig_scatter, ax_scatter = plt.subplots(figsize=(12, 7))
        scatter = ax_scatter.scatter(
            association_rules_pd['support'], 
            association_rules_pd['confidence'],
            c=association_rules_pd['lift'],
            s=association_rules_pd['lift'] * 50,
            cmap='viridis',
            alpha=0.6,
            edgecolors='white',
            linewidth=0.5
        )
        
        ax_scatter.set_title('Support vs Confidence for Association Rules', fontsize=16, fontweight='bold')
        ax_scatter.set_xlabel('Support', fontsize=12)
        ax_scatter.set_ylabel('Confidence', fontsize=12)
        ax_scatter.grid(True, linestyle='--', alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_scatter)
        cbar.set_label('Lift', fontsize=12)
        
        st.pyplot(fig_scatter)
        plt.close()
    else:
        st.warning("No association rules found to plot.")
    
    # Top Rules by Lift
    st.markdown("### 📊 Top 10 Rules by Lift")
    
    if not association_rules_pd.empty:
        top_rules_by_lift = association_rules_pd.nlargest(10, 'lift').copy()
        top_rules_by_lift['rule_str'] = top_rules_by_lift.apply(
            lambda row: f"{', '.join(row['antecedent'])} → {', '.join(row['consequent'])}", 
            axis=1
        )
        
        fig_lift, ax_lift = plt.subplots(figsize=(12, 7))
        colors = sns.color_palette("magma", n_colors=len(top_rules_by_lift))
        bars = ax_lift.barh(top_rules_by_lift['rule_str'], top_rules_by_lift['lift'], color=colors)
        ax_lift.set_title('Top Association Rules by Lift', fontsize=16, fontweight='bold')
        ax_lift.set_xlabel('Lift', fontsize=12)
        ax_lift.set_ylabel('Association Rule', fontsize=12)
        ax_lift.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax_lift.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', ha='left', va='center', fontsize=10)
        
        st.pyplot(fig_lift)
        plt.close()
    else:
        st.warning("No association rules found to display.")

# ============================================================================
# TAB 3: RULE EXPLORER
# ============================================================================

with tab3:
    st.markdown("### 🔍 Interactive Rule Explorer")
    
    if not association_rules_pd.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_lift_filter = st.slider(
                "Minimum Lift", 
                min_value=0.0, 
                max_value=float(association_rules_pd['lift'].max()), 
                value=1.0,
                step=0.1
            )
        
        with col2:
            min_support_filter = st.slider(
                "Minimum Support", 
                min_value=0.0, 
                max_value=float(association_rules_pd['support'].max()), 
                value=float(association_rules_pd['support'].min()),
                step=0.001
            )
        
        with col3:
            search_item = st.text_input("🔎 Search for item", placeholder="e.g., milk, bread, eggs")
        
        # Apply filters
        filtered_rules = association_rules_pd.copy()
        
        if min_lift_filter > 0:
            filtered_rules = filtered_rules[filtered_rules['lift'] >= min_lift_filter]
        
        if min_support_filter > 0:
            filtered_rules = filtered_rules[filtered_rules['support'] >= min_support_filter]
        
        if search_item:
            search_lower = search_item.lower()
            filtered_rules = filtered_rules[
                filtered_rules['antecedent'].apply(
                    lambda x: any(search_lower in item for item in x) if isinstance(x, list) else search_lower in str(x)
                ) |
                filtered_rules['consequent'].apply(
                    lambda x: any(search_lower in item for item in x) if isinstance(x, list) else search_lower in str(x)
                )
            ]
        
        st.caption(f"Showing {len(filtered_rules)} rules out of {len(association_rules_pd)}")
        
        if not filtered_rules.empty:
            # Display filtered rules
            display_filtered = filtered_rules.copy()
            display_filtered['antecedent_str'] = display_filtered['antecedent'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
            display_filtered['consequent_str'] = display_filtered['consequent'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
            
            st.dataframe(
                display_filtered[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].rename(
                    columns={
                        'antecedent_str': 'Antecedent',
                        'consequent_str': 'Consequent',
                        'support': 'Support',
                        'confidence': 'Confidence',
                        'lift': 'Lift'
                    }
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Rule Details
            st.markdown("### 📖 Rule Details")
            
            # Create rule strings for selectbox
            rule_strings = display_filtered.apply(
                lambda x: f"{', '.join(x['antecedent'])} → {', '.join(x['consequent'])}", 
                axis=1
            ).tolist()
            
            selected_rule = st.selectbox("Select a rule to view details", rule_strings)
            
            if selected_rule:
                rule_idx = rule_strings.index(selected_rule)
                rule = display_filtered.iloc[rule_idx]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Support", f"{rule['support']:.4f}")
                    st.caption("Fraction of transactions containing both itemsets")
                
                with col2:
                    st.metric("🎯 Confidence", f"{rule['confidence']:.4f}")
                    st.caption("Probability of consequent given antecedent")
                
                with col3:
                    st.metric("📈 Lift", f"{rule['lift']:.4f}")
                    st.caption("How much more likely consequent occurs with antecedent")
                
                # Interpretation
                if rule['lift'] > 1:
                    lift_interpretation = f"Customers who buy **{', '.join(rule['antecedent'])}** are **{rule['lift']:.1f}x** more likely to also buy **{', '.join(rule['consequent'])}** compared to random chance."
                elif rule['lift'] == 1:
                    lift_interpretation = f"Items **{', '.join(rule['antecedent'])}** and **{', '.join(rule['consequent'])}** are independent of each other."
                else:
                    lift_interpretation = f"Customers who buy **{', '.join(rule['antecedent'])}** are less likely to buy **{', '.join(rule['consequent'])}**."
                
                st.info(f"💡 **Insight:** {lift_interpretation}")
                
                st.caption(f"Confidence means: {rule['confidence']:.1%} of customers who bought {', '.join(rule['antecedent'])} also bought {', '.join(rule['consequent'])}")
        else:
            st.warning("No rules match the current filters. Try adjusting the parameters.")
    else:
        st.warning("No association rules found. Try lowering the minimum support or confidence.")

# ============================================================================
# TAB 4: NETWORK GRAPH
# ============================================================================

with tab4:
    st.markdown("### 🌐 Interactive Association Rule Network")
    
    if not association_rules_pd.empty:
        # Network settings
        col1, col2 = st.columns(2)
        
        with col1:
            max_nodes = st.slider("Maximum number of rules to display", 
                                 min_value=5, max_value=50, value=20, step=5)
        
        with col2:
            layout_option = st.selectbox(
                "Network Layout",
                ["Spring", "Circular", "Random"],
                help="Different layouts for visualizing the network"
            )
        
        # Create network graph
        G = nx.DiGraph()
        
        # Use top rules
        top_rules_for_network = association_rules_pd.nlargest(max_nodes, 'lift')
        
        for _, row in top_rules_for_network.iterrows():
            antecedent = row['antecedent']
            consequent = row['consequent']
            
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
            
            G.add_edge(antecedent_str, consequent_str, 
                      weight=row['lift'], 
                      confidence=row['confidence'],
                      support=row['support'])
        
        # Layout
        if layout_option == "Spring":
            pos = nx.spring_layout(G, k=0.5, iterations=100, scale=2, seed=42)
        elif layout_option == "Circular":
            pos = nx.circular_layout(G, scale=2)
        else:
            pos = nx.random_layout(G, seed=42)
        
        # Create Plotly figure
        plotly_fig = go.Figure()
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            if G.nodes[node].get('type') == 'item':
                node_colors.append('lightblue')
                node_sizes.append(25)
            else:
                node_colors.append('lightgreen')
                node_sizes.append(35)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            hovertext=node_text
        )
        
        plotly_fig.add_trace(node_trace)
        
        # Add edges
        plotly_cmap = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', 
                      '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
        
        all_conf_values = [d['confidence'] for _, _, d in G.edges(data=True)]
        min_conf = min(all_conf_values) if all_conf_values else 0
        max_conf = max(all_conf_values) if all_conf_values else 1
        
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            current_conf = d['confidence']
            normalized_conf = (current_conf - min_conf) / (max_conf - min_conf) if max_conf - min_conf > 0 else 0.5
            color_idx = int(normalized_conf * (len(plotly_cmap) - 1))
            edge_color = plotly_cmap[color_idx]
            
            hover_text = f"{u} → {v}<br>Lift: {d['weight']:.2f}<br>Confidence: {d['confidence']:.2f}<br>Support: {d['support']:.3f}"
            
            plotly_fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=d['weight'] * 2.5, color=edge_color),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))
        
        plotly_fig.update_layout(
            title=f'Association Rule Network (Top {max_nodes} Rules)',
            title_x=0.5,
            title_font_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(plotly_fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        <div style="display: flex; gap: 20px; justify-content: center; padding: 10px;">
            <span><span style="display: inline-block; width: 15px; height: 15px; background: lightblue; border-radius: 50%;"></span> Individual Item</span>
            <span><span style="display: inline-block; width: 15px; height: 15px; background: lightgreen; border-radius: 50%;"></span> Itemset (Group)</span>
            <span><span style="display: inline-block; width: 30px; height: 4px; background: #b2182b;"></span> Edge Color = Confidence</span>
            <span><span style="display: inline-block; width: 30px; height: 4px; background: #2166ac;"></span> Edge Width = Lift</span>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.warning("No association rules found to generate network graph.")

# ============================================================================
# TAB 5: ADVANCED ANALYTICS
# ============================================================================

with tab5:
    st.markdown("### 📈 Advanced Analytics")
    
    # Get top items first (needed for multiple visualizations)
    exploded_items = final_baskets_df.withColumn("item", F.explode("items"))
    item_counts = exploded_items.groupBy("item").count().orderBy(F.desc("count"))
    top_items_pd = item_counts.limit(20).toPandas()
    
    # Distribution of Lift Values
    if not association_rules_pd.empty:
        st.markdown("#### 📊 Distribution of Lift Values")
        
        fig_lift_dist, ax_lift_dist = plt.subplots(figsize=(10, 6))
        ax_lift_dist.hist(association_rules_pd['lift'], bins=20, color='#6B73FF', edgecolor='white', alpha=0.7)
        ax_lift_dist.axvline(association_rules_pd['lift'].mean(), color='red', linestyle='--', 
                            label=f'Mean: {association_rules_pd["lift"].mean():.2f}')
        ax_lift_dist.axvline(association_rules_pd['lift'].median(), color='green', linestyle='--', 
                            label=f'Median: {association_rules_pd["lift"].median():.2f}')
        ax_lift_dist.set_title('Distribution of Lift Values', fontsize=16, fontweight='bold')
        ax_lift_dist.set_xlabel('Lift', fontsize=12)
        ax_lift_dist.set_ylabel('Frequency', fontsize=12)
        ax_lift_dist.legend()
        ax_lift_dist.grid(axis='y', alpha=0.3)
        st.pyplot(fig_lift_dist)
        plt.close()
    
    # 1. Co-occurrence Matrix
    st.markdown("#### 🔥 Top Items Co-occurrence Matrix")
    
    top_n_cooccur = st.slider("Number of top items", min_value=5, max_value=20, value=10, step=1)
    
    top_items_list = top_items_pd['item'].head(top_n_cooccur).tolist()
    
    # Create co-occurrence matrix
    co_occurrence = pd.DataFrame(0, index=top_items_list, columns=top_items_list)
    
    for items in final_baskets_df_pd['items']:
        present = [item for item in top_items_list if item in items]
        for i, item1 in enumerate(present):
            for item2 in present[i+1:]:
                co_occurrence.loc[item1, item2] += 1
                co_occurrence.loc[item2, item1] += 1
    
    # Plot heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        co_occurrence, 
        annot=True, 
        fmt='d', 
        cmap='YlOrRd', 
        square=True, 
        linewidths=0.5,
        cbar_kws={'label': 'Co-occurrence Count'},
        ax=ax_heatmap
    )
    ax_heatmap.set_title(f'Top {top_n_cooccur} Items Co-occurrence Matrix', fontsize=16, fontweight='bold')
    ax_heatmap.set_xlabel('Items', fontsize=12)
    ax_heatmap.set_ylabel('Items', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig_heatmap)
    plt.close()
    
    # 2. Additional Analytics: Item Pair Frequencies
    st.markdown("#### 📊 Top Item Pairs")
    
    pair_counts = []
    for i, item1 in enumerate(top_items_list):
        for item2 in top_items_list[i+1:]:
            count = co_occurrence.loc[item1, item2]
            if count > 0:
                pair_counts.append({'Item 1': item1, 'Item 2': item2, 'Frequency': count})
    
    pair_df = pd.DataFrame(pair_counts).nlargest(10, 'Frequency')
    
    if not pair_df.empty:
        fig_pairs, ax_pairs = plt.subplots(figsize=(10, 6))
        pair_df['Pair'] = pair_df['Item 1'] + ' & ' + pair_df['Item 2']
        bars = ax_pairs.barh(pair_df['Pair'], pair_df['Frequency'], 
                            color=sns.color_palette("viridis", n_colors=len(pair_df)))
        ax_pairs.set_title('Top 10 Most Frequent Item Pairs', fontsize=16, fontweight='bold')
        ax_pairs.set_xlabel('Co-occurrence Frequency', fontsize=12)
        ax_pairs.set_ylabel('Item Pair', fontsize=12)
        ax_pairs.grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            ax_pairs.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{int(width)}', ha='left', va='center', fontsize=10)
        
        st.pyplot(fig_pairs)
        plt.close()
    else:
        st.info("No item pairs found in the top items list.")
    
    # 3. Rule Distribution Statistics
    if not association_rules_pd.empty:
        st.markdown("#### 📊 Rule Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Lift", f"{association_rules_pd['lift'].mean():.2f}")
            st.caption(f"Min: {association_rules_pd['lift'].min():.2f} | Max: {association_rules_pd['lift'].max():.2f}")
        
        with col2:
            st.metric("Average Confidence", f"{association_rules_pd['confidence'].mean():.2f}")
            st.caption(f"Min: {association_rules_pd['confidence'].min():.2f} | Max: {association_rules_pd['confidence'].max():.2f}")
        
        with col3:
            st.metric("Average Support", f"{association_rules_pd['support'].mean():.3f}")
            st.caption(f"Min: {association_rules_pd['support'].min():.3f} | Max: {association_rules_pd['support'].max():.3f}")
        
        # Distribution plots
        fig_stats, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confidence distribution
        ax1.hist(association_rules_pd['confidence'], bins=20, color='#00D2FF', edgecolor='white', alpha=0.7)
        ax1.axvline(association_rules_pd['confidence'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {association_rules_pd["confidence"].mean():.2f}')
        ax1.set_title('Distribution of Confidence Values', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Support distribution
        ax2.hist(association_rules_pd['support'], bins=20, color='#FF6B6B', edgecolor='white', alpha=0.7)
        ax2.axvline(association_rules_pd['support'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {association_rules_pd["support"].mean():.3f}')
        ax2.set_title('Distribution of Support Values', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Support', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_stats)
        plt.close()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>BasketSense v1.0 | Built with PySpark & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
