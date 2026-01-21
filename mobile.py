import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Mobile Analytics of 2025",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DATA LOADING & CLEANING
# =====================================================
@st.cache_data
def load_and_clean_data():
    """Load and clean the mobile dataset with comprehensive preprocessing"""
    try:
        df = pd.read_csv("Mobiles Dataset (2025).csv", encoding='cp1252')
        
        # Rename columns
        df = df.rename(columns={
            'Company Name': 'Brand',
            'Model Name': 'Model',
            'Mobile Weight': 'Weight_g',
            'RAM': 'RAM_GB',
            'Battery Capacity': 'Battery_mAh',
            'Screen Size': 'Screen_Size_Inches',
            'Launched Price (Pakistan)': 'Price_Pakistan',
            'Launched Price (India)': 'Price_India',
            'Launched Price (China)': 'Price_China',
            'Launched Price (USA)': 'Price_USA',
            'Launched Price (Dubai)': 'Price_Dubai',
            'Launched Year': 'Launch_Year'
        })
        
        # Clean numeric columns
        df['Weight_g'] = df['Weight_g'].astype(str).str.extract('(\d+\.?\d*)')[0]
        df['Weight_g'] = pd.to_numeric(df['Weight_g'], errors='coerce')
        
        df['RAM_GB'] = df['RAM_GB'].astype(str).str.extract('(\d+\.?\d*)')[0]
        df['RAM_GB'] = pd.to_numeric(df['RAM_GB'], errors='coerce')
        
        df['Battery_mAh'] = df['Battery_mAh'].astype(str).str.extract('(\d+\.?\d*)')[0]
        df['Battery_mAh'] = pd.to_numeric(df['Battery_mAh'], errors='coerce')
        
        df['Screen_Size_Inches'] = df['Screen_Size_Inches'].astype(str).str.extract('(\d+\.?\d*)')[0]
        df['Screen_Size_Inches'] = pd.to_numeric(df['Screen_Size_Inches'], errors='coerce')
        
        # Clean price columns
        price_columns = ['Price_Pakistan', 'Price_India', 'Price_China', 'Price_USA', 'Price_Dubai']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Launch_Year'] = pd.to_numeric(df['Launch_Year'], errors='coerce')
        
        df = df.dropna(subset=['Brand', 'Model'])
        df = df.drop_duplicates(subset=['Brand', 'Model'], keep='first')
        
        return df
        
    except FileNotFoundError:
        st.error("❌ Dataset file 'Mobiles Dataset 2025.csv' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

df = load_and_clean_data()

# =====================================================
# AI FUNCTIONS - SMART ANALYSIS
# =====================================================

def calculate_value_score(row, price_col):
    """AI-based value scoring system"""
    score = 0
    price = row[price_col]
    
    if pd.isna(price) or price == 0:
        return 0
    
    # RAM value (higher is better)
    if not pd.isna(row['RAM_GB']):
        score += (row['RAM_GB'] / price) * 100000
    
    # Battery value
    if not pd.isna(row['Battery_mAh']):
        score += (row['Battery_mAh'] / price) * 20
    
    # Screen size value
    if not pd.isna(row['Screen_Size_Inches']):
        score += (row['Screen_Size_Inches'] / price) * 10000
    
    return score

def predict_fair_price(row, df_subset, price_col):
    """Simple price prediction based on specifications"""
    if df_subset.empty or len(df_subset) < 5:
        return None
    
    df_valid = df_subset.dropna(subset=[price_col, 'RAM_GB', 'Battery_mAh'])
    
    if len(df_valid) < 5:
        return None
    
    # Calculate average price per GB RAM
    avg_price_per_ram = df_valid[price_col].sum() / df_valid['RAM_GB'].sum()
    
    # Calculate average price per mAh battery
    avg_price_per_battery = df_valid[price_col].sum() / df_valid['Battery_mAh'].sum()
    
    # Predict price
    predicted = 0
    if not pd.isna(row['RAM_GB']) and row['RAM_GB'] > 0:
        predicted += row['RAM_GB'] * avg_price_per_ram * 0.6
    if not pd.isna(row['Battery_mAh']) and row['Battery_mAh'] > 0:
        predicted += row['Battery_mAh'] * avg_price_per_battery * 0.4
    
    return predicted if predicted > 0 else None

def find_market_segments(df, price_col):
    """Automatic market segmentation"""
    df_clean = df.dropna(subset=[price_col])
    
    if len(df_clean) == 0:
        return df_clean
    
    q1 = df_clean[price_col].quantile(0.33)
    q2 = df_clean[price_col].quantile(0.67)
    
    segments = []
    for _, row in df_clean.iterrows():
        price = row[price_col]
        if price <= q1:
            segments.append('Budget')
        elif price <= q2:
            segments.append('Mid-Range')
        else:
            segments.append('Premium')
    
    df_clean['Segment'] = segments
    return df_clean

def detect_outliers(df, price_col):
    """Detect best value and overpriced phones"""
    df_clean = df.dropna(subset=[price_col, 'RAM_GB', 'Battery_mAh'])
    
    if len(df_clean) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate value scores
    df_clean['Value_Score'] = df_clean.apply(lambda x: calculate_value_score(x, price_col), axis=1)
    
    best_value = df_clean.nlargest(5, 'Value_Score')[['Brand', 'Model', 'RAM_GB', 'Battery_mAh', price_col, 'Value_Score']]
    overpriced = df_clean.nsmallest(5, 'Value_Score')[['Brand', 'Model', 'RAM_GB', 'Battery_mAh', price_col, 'Value_Score']]
    
    return best_value, overpriced

# =====================================================
# THEME CONFIGURATION - FIXED FOR LIGHT MODE
# =====================================================
def apply_theme(theme_mode):
    if theme_mode == "Dark Mode":
        bg_color = "#0E1117"
        text_color = "#FAFAFA"
        card_bg = "#1E2127"
        border_color = "#4A4A4A"
        card_text = "#FFFFFF"
    else:
        bg_color = "#FFFFFF"
        text_color = "#000000"
        card_bg = "#F0F2F6"
        border_color = "#CCCCCC"
        card_text = "#1a1a1a"
    
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .recommendation-card {{
            background-color: {card_bg};
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            margin: 10px 0;
            color: {card_text} !important;
        }}
        .recommendation-card h3 {{
            color: {card_text} !important;
            margin: 0 0 10px 0;
        }}
        .recommendation-card p {{
            color: {card_text} !important;
            margin: 5px 0;
        }}
        .recommendation-card strong {{
            color: {card_text} !important;
        }}
        .insight-card {{
            background-color: {card_bg};
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2196F3;
            margin: 10px 0;
            color: {card_text} !important;
        }}
        .insight-card h4 {{
            color: {card_text} !important;
            margin: 0 0 10px 0;
        }}
        .insight-card p {{
            color: {card_text} !important;
            margin: 5px 0;
        }}
        .insight-card strong {{
            color: {card_text} !important;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    return bg_color, text_color, card_bg

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Dashboard Controls")

# Theme
theme_mode = st.sidebar.radio("🎨 Theme", ["Light Mode", "Dark Mode"], index=1)
bg_color, text_color, card_bg = apply_theme(theme_mode)

# Region
st.sidebar.markdown("---")
price_region = st.sidebar.selectbox("🌍 Price Region", 
    ["Pakistan", "India", "China", "USA", "Dubai"], index=0)
price_column = f"Price_{price_region}"

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Navigation")
page = st.sidebar.radio("Go to:", [
    "📊 Dashboard Overview",
    "🤖 AI Recommendations", 
    "💡 Smart Insights",
    "🎯 Price Predictor",
    "📈 Market Analysis"
])

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Data Filters")

available_brands = sorted(df['Brand'].dropna().unique())
selected_brands = st.sidebar.multiselect("Brand", options=available_brands,
    default=available_brands[:5] if len(available_brands) >= 5 else available_brands)

if 'Launch_Year' in df.columns:
    year_min, year_max = int(df['Launch_Year'].min()), int(df['Launch_Year'].max())
    year_range = st.sidebar.slider("Launch Year", year_min, year_max, (year_min, year_max))

if 'RAM_GB' in df.columns:
    ram_min, ram_max = float(df['RAM_GB'].min()), float(df['RAM_GB'].max())
    ram_range = st.sidebar.slider("RAM (GB)", ram_min, ram_max, (ram_min, ram_max))

if 'Battery_mAh' in df.columns:
    battery_min = float(df['Battery_mAh'].min())
    battery_max = float(df['Battery_mAh'].max())
    battery_range = st.sidebar.slider("Battery (mAh)", battery_min, battery_max, (battery_min, battery_max))

if price_column in df.columns:
    price_min, price_max = float(df[price_column].min()), float(df[price_column].max())
    price_range = st.sidebar.slider(f"Price ({price_region})", price_min, price_max, (price_min, price_max))

# Apply filters
filtered_df = df.copy()
if selected_brands:
    filtered_df = filtered_df[filtered_df['Brand'].isin(selected_brands)]
if 'Launch_Year' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['Launch_Year'] >= year_range[0]) & 
                              (filtered_df['Launch_Year'] <= year_range[1])]
if 'RAM_GB' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['RAM_GB'] >= ram_range[0]) & 
                              (filtered_df['RAM_GB'] <= ram_range[1])]
if 'Battery_mAh' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['Battery_mAh'] >= battery_range[0]) & 
                              (filtered_df['Battery_mAh'] <= battery_range[1])]
if price_column in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df[price_column] >= price_range[0]) & 
                              (filtered_df[price_column] <= price_range[1])]

# =====================================================
# PAGE 1: DASHBOARD OVERVIEW
# =====================================================
if page == "📊 Dashboard Overview":
    st.title("📱 AI-Powered Mobile Analytics Dashboard 2025")
    st.markdown(f"**Region:** {price_region} | **Models:** {len(filtered_df):,} of {len(df):,}")
    st.markdown("---")
    
    # KPIs
    st.subheader("📊 Executive Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Models", f"{len(filtered_df):,}")
    with col2:
        if price_column in filtered_df.columns:
            st.metric("Avg Price", f"{filtered_df[price_column].mean():,.0f}")
    with col3:
        if price_column in filtered_df.columns:
            st.metric("Median Price", f"{filtered_df[price_column].median():,.0f}")
    with col4:
        if 'RAM_GB' in filtered_df.columns:
            st.metric("Avg RAM", f"{filtered_df['RAM_GB'].mean():.1f} GB")
    with col5:
        if 'Battery_mAh' in filtered_df.columns:
            st.metric("Avg Battery", f"{filtered_df['Battery_mAh'].mean():.0f} mAh")
    with col6:
        if len(filtered_df) > 0:
            st.metric("Top Brand", filtered_df['Brand'].value_counts().index[0])
    
    st.markdown("---")
    
    # Brand Analysis
    st.subheader("🏢 Brand Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if price_column in filtered_df.columns:
            brand_price = filtered_df.groupby('Brand')[price_column].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(brand_price.index, brand_price.values, color='#1f77b4')
            ax.set_xlabel(f'Avg Price ({price_region})', fontsize=12)
            ax.set_title('Top 10 Brands by Price', fontsize=14, fontweight='bold')
            if theme_mode == "Dark Mode":
                ax.set_facecolor('#1E2127')
                fig.patch.set_facecolor('#1E2127')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        brand_count = filtered_df['Brand'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(brand_count.index, brand_count.values, color='#ff7f0e')
        ax.set_xlabel('Model Count', fontsize=12)
        ax.set_title('Top 10 Brands by Models', fontsize=14, fontweight='bold')
        if theme_mode == "Dark Mode":
            ax.set_facecolor('#1E2127')
            fig.patch.set_facecolor('#1E2127')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Scatter Plots
    st.subheader("📈 Specifications vs Price")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'RAM_GB' in filtered_df.columns and price_column in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            for brand in filtered_df['Brand'].value_counts().head(8).index:
                data = filtered_df[filtered_df['Brand'] == brand]
                ax.scatter(data['RAM_GB'], data[price_column], label=brand, alpha=0.6, s=60)
            ax.set_xlabel('RAM (GB)', fontsize=12)
            ax.set_ylabel(f'Price ({price_region})', fontsize=12)
            ax.set_title('RAM vs Price', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
            ax.grid(True, alpha=0.3)
            if theme_mode == "Dark Mode":
                ax.set_facecolor('#1E2127')
                fig.patch.set_facecolor('#1E2127')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.grid(color='gray', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        if 'Battery_mAh' in filtered_df.columns and price_column in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            for brand in filtered_df['Brand'].value_counts().head(8).index:
                data = filtered_df[filtered_df['Brand'] == brand]
                ax.scatter(data['Battery_mAh'], data[price_column], label=brand, alpha=0.6, s=60)
            ax.set_xlabel('Battery (mAh)', fontsize=12)
            ax.set_ylabel(f'Price ({price_region})', fontsize=12)
            ax.set_title('Battery vs Price', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
            ax.grid(True, alpha=0.3)
            if theme_mode == "Dark Mode":
                ax.set_facecolor('#1E2127')
                fig.patch.set_facecolor('#1E2127')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.grid(color='gray', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)

# =====================================================
# PAGE 2: AI RECOMMENDATIONS - FIXED
# =====================================================
elif page == "🤖 AI Recommendations":
    st.title("🤖 AI-Powered Phone Recommender")
    st.markdown("Get personalized phone recommendations based on your preferences")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_budget = st.number_input(f"Your Budget ({price_region})", 
            min_value=0, max_value=int(df[price_column].max()), 
            value=int(df[price_column].median()), step=5000)
    
    with col2:
        user_ram = st.selectbox("Minimum RAM", [4, 6, 8, 12, 16])
    
    with col3:
        user_battery = st.selectbox("Minimum Battery (mAh)", [3000, 4000, 5000, 6000])
    
    if st.button("🔍 Find Best Phones", type="primary"):
        # FIXED: Use original df instead of filtered_df to ignore sidebar filters
        recommendations = df[
            (df[price_column] <= user_budget) &
            (df[price_column] > 0) &
            (df['RAM_GB'] >= user_ram) &
            (df['Battery_mAh'] >= user_battery)
        ].copy()
        
        # Remove any rows with missing critical data
        recommendations = recommendations.dropna(subset=[price_column, 'RAM_GB', 'Battery_mAh', 'Screen_Size_Inches'])
        
        if len(recommendations) > 0:
            # Calculate value scores
            recommendations['Value_Score'] = recommendations.apply(
                lambda x: calculate_value_score(x, price_column), axis=1)
            
            # Get top 5
            top_recommendations = recommendations.nlargest(5, 'Value_Score')
            
            st.success(f"✅ Found {len(recommendations)} phones matching your criteria. Here are the top 5 best values:")
            st.markdown("---")
            
            for idx, (_, phone) in enumerate(top_recommendations.iterrows(), 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>#{idx} {phone['Brand']} {phone['Model']}</h3>
                    <p><strong>💰 Price:</strong> {phone[price_column]:,.0f} {price_region}</p>
                    <p><strong>🧠 RAM:</strong> {phone['RAM_GB']:.0f} GB | 
                       <strong>🔋 Battery:</strong> {phone['Battery_mAh']:.0f} mAh | 
                       <strong>📱 Screen:</strong> {phone['Screen_Size_Inches']:.1f}"</p>
                    <p><strong>⭐ AI Value Score:</strong> {phone['Value_Score']:.2f}/100</p>
                    <p style="font-size: 0.9em; margin-top: 10px;">This phone offers excellent specifications for the price point, making it a smart purchase in the {price_region} market.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"❌ No phones found with Budget ≤ {user_budget:,}, RAM ≥ {user_ram}GB, Battery ≥ {user_battery}mAh in the {price_region} market.")
            st.info("💡 **Tip:** Try increasing your budget or reducing RAM/Battery requirements to see more options.")

# =====================================================
# PAGE 3: SMART INSIGHTS
# =====================================================
elif page == "💡 Smart Insights":
    st.title("💡 AI-Generated Market Insights")
    st.markdown("Automatic pattern discovery and intelligent analysis")
    st.markdown("---")
    
    st.subheader("🎯 Best Value Phones (AI Detected)")
    best_value, overpriced = detect_outliers(filtered_df, price_column)
    
    if len(best_value) > 0:
        st.success("These phones offer exceptional value for money:")
        st.dataframe(best_value.style.format({
            price_column: '{:,.0f}',
            'RAM_GB': '{:.0f} GB',
            'Battery_mAh': '{:.0f} mAh',
            'Value_Score': '{:.2f}'
        }), use_container_width=True)
    else:
        st.info("Not enough data to calculate value scores")
    
    st.markdown("---")
    st.subheader("⚠️ Overpriced Phones (AI Detected)")
    
    if len(overpriced) > 0:
        st.warning("These phones are expensive relative to their specifications:")
        st.dataframe(overpriced.style.format({
            price_column: '{:,.0f}',
            'RAM_GB': '{:.0f} GB',
            'Battery_mAh': '{:.0f} mAh',
            'Value_Score': '{:.2f}'
        }), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Automated Pattern Discovery")
    
    # Pattern 1: RAM vs Price correlation
    if 'RAM_GB' in filtered_df.columns and price_column in filtered_df.columns:
        high_ram = filtered_df[filtered_df['RAM_GB'] >= 12]
        low_ram = filtered_df[filtered_df['RAM_GB'] < 12]
        if len(high_ram) > 0 and len(low_ram) > 0:
            high_avg = high_ram[price_column].mean()
            low_avg = low_ram[price_column].mean()
            diff_pct = ((high_avg - low_avg) / low_avg) * 100
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>🧠 RAM Analysis</h4>
                <p>Phones with ≥12GB RAM cost <strong>{diff_pct:.1f}% more</strong> on average 
                ({high_avg:,.0f} vs {low_avg:,.0f})</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Pattern 2: Battery analysis
    if 'Battery_mAh' in filtered_df.columns:
        high_battery = filtered_df[filtered_df['Battery_mAh'] >= 5000]
        if len(high_battery) > 0:
            pct_high_battery = (len(high_battery) / len(filtered_df)) * 100
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>🔋 Battery Trend</h4>
                <p><strong>{pct_high_battery:.1f}%</strong> of phones now feature ≥5000mAh batteries, 
                showing the market shift toward longer battery life</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Pattern 3: Brand dominance
    if len(filtered_df) > 0:
        top_brand = filtered_df['Brand'].value_counts().index[0]
        top_count = filtered_df['Brand'].value_counts().values[0]
        market_share = (top_count / len(filtered_df)) * 100
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>🏢 Market Leader</h4>
            <p><strong>{top_brand}</strong> dominates with {top_count} models 
            ({market_share:.1f}% market share in this dataset)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pattern 4: Price distribution
    if price_column in filtered_df.columns:
        cheap_phones = filtered_df[filtered_df[price_column] < filtered_df[price_column].quantile(0.33)]
        expensive_phones = filtered_df[filtered_df[price_column] > filtered_df[price_column].quantile(0.67)]
        
        if len(cheap_phones) > 0 and len(expensive_phones) > 0:
            cheap_avg_ram = cheap_phones['RAM_GB'].mean()
            expensive_avg_ram = expensive_phones['RAM_GB'].mean()
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>💎 Premium vs Budget Specs</h4>
                <p>Premium phones (top 33% by price) have <strong>{expensive_avg_ram:.1f}GB</strong> RAM on average, 
                while budget phones have <strong>{cheap_avg_ram:.1f}GB</strong> RAM</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# PAGE 4: PRICE PREDICTOR
# =====================================================
elif page == "🎯 Price Predictor":
    st.title("🎯 AI Price Prediction Engine")
    st.markdown("Predict fair prices and find over/underpriced phones")
    st.markdown("---")
    
    st.subheader("📊 Price vs Predicted Price Analysis")
    
    # Calculate predictions for all phones
    df_with_predictions = filtered_df.dropna(subset=[price_column, 'RAM_GB', 'Battery_mAh']).copy()
    
    if len(df_with_predictions) >= 5:
        predictions = []
        for _, row in df_with_predictions.iterrows():
            pred = predict_fair_price(row, df_with_predictions, price_column)
            predictions.append(pred)
        
        df_with_predictions['Predicted_Price'] = predictions
        df_with_predictions = df_with_predictions.dropna(subset=['Predicted_Price'])
        
        if len(df_with_predictions) > 0:
            df_with_predictions['Price_Difference'] = df_with_predictions[price_column] - df_with_predictions['Predicted_Price']
            df_with_predictions['Price_Diff_Pct'] = (df_with_predictions['Price_Difference'] / df_with_predictions['Predicted_Price']) * 100
            
            # Show underpriced phones
            st.success("🎉 Best Deals - Underpriced Phones")
            underpriced = df_with_predictions[df_with_predictions['Price_Diff_Pct'] < -10].nsmallest(10, 'Price_Diff_Pct')
            
            if len(underpriced) > 0:
                st.dataframe(underpriced[['Brand', 'Model', price_column, 'Predicted_Price', 'Price_Diff_Pct']].style.format({
                    price_column: '{:,.0f}',
                    'Predicted_Price': '{:,.0f}',
                    'Price_Diff_Pct': '{:.1f}%'
                }), use_container_width=True)
            else:
                st.info("No significantly underpriced phones found in current filters")
            
            st.markdown("---")
            
            # Show overpriced phones
            st.warning("⚠️ Overpriced Phones")
            overpriced_phones = df_with_predictions[df_with_predictions['Price_Diff_Pct'] > 10].nlargest(10, 'Price_Diff_Pct')
            
            if len(overpriced_phones) > 0:
                st.dataframe(overpriced_phones[['Brand', 'Model', price_column, 'Predicted_Price', 'Price_Diff_Pct']].style.format({
                    price_column: '{:,.0f}',
                    'Predicted_Price': '{:,.0f}',
                    'Price_Diff_Pct': '{:.1f}%'
                }), use_container_width=True)
            else:
                st.info("No significantly overpriced phones found")
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Actual vs Predicted Price Comparison")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(df_with_predictions['Predicted_Price'], df_with_predictions[price_column], alpha=0.6, s=50)
            
            # Perfect prediction line
            max_val = max(df_with_predictions[price_column].max(), df_with_predictions['Predicted_Price'].max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel('Predicted Price', fontsize=12)
            ax.set_ylabel(f'Actual Price ({price_region})', fontsize=12)
            ax.set_title('Price Prediction Accuracy', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if theme_mode == "Dark Mode":
                ax.set_facecolor('#1E2127')
                fig.patch.set_facecolor('#1E2127')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.grid(color='gray', alpha=0.2)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Not enough data for price prediction")
    else:
        st.warning("Need at least 5 phones to perform price prediction. Please adjust your filters.")

# =====================================================
# PAGE 5: MARKET ANALYSIS
# =====================================================
elif page == "📈 Market Analysis":
    st.title("📈 Advanced Market Segmentation")
    st.markdown("AI-powered market segment discovery and analysis")
    st.markdown("---")
    
    # Market Segmentation
    segmented_df = find_market_segments(filtered_df, price_column)
    
    if len(segmented_df) > 0:
        st.subheader("🎯 Market Segments Distribution")
        
        segment_counts = segmented_df['Segment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            ax.set_title('Market Share by Segment', fontsize=14, fontweight='bold')
            
            if theme_mode == "Dark Mode":
                fig.patch.set_facecolor('#1E2127')
                ax.title.set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Segment Statistics")
            for segment in ['Budget', 'Mid-Range', 'Premium']:
                seg_data = segmented_df[segmented_df['Segment'] == segment]
                if len(seg_data) > 0:
                    avg_price = seg_data[price_column].mean()
                    avg_ram = seg_data['RAM_GB'].mean()
                    count = len(seg_data)
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{segment}</h4>
                        <p><strong>Models:</strong> {count} | <strong>Avg Price:</strong> {avg_price:,.0f}</p>
                        <p><strong>Avg RAM:</strong> {avg_ram:.1f} GB</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Segment-wise brand analysis
        st.subheader("🏢 Brand Performance Across Segments")
        
        segment_brand = segmented_df.groupby(['Segment', 'Brand']).size().reset_index(name='Count')
        pivot_data = segment_brand.pivot(index='Brand', columns='Segment', values='Count').fillna(0)
        
        if len(pivot_data) > 0:
            top_brands_seg = pivot_data.sum(axis=1).nlargest(10).index
            pivot_data_top = pivot_data.loc[top_brands_seg]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot_data_top.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff', '#99ff99'])
            ax.set_xlabel('Brand', fontsize=12)
            ax.set_ylabel('Number of Models', fontsize=12)
            ax.set_title('Brand Distribution Across Market Segments', fontsize=14, fontweight='bold')
            ax.legend(title='Segment')
            plt.xticks(rotation=45, ha='right')
            
            if theme_mode == "Dark Mode":
                ax.set_facecolor('#1E2127')
                fig.patch.set_facecolor('#1E2127')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Correlation Heatmap
        st.subheader("🔥 Feature Correlation Matrix")
        
        corr_cols = ['RAM_GB', 'Battery_mAh', 'Screen_Size_Inches', 'Weight_g', price_column]
        available_corr = [c for c in corr_cols if c in segmented_df.columns]
        
        if len(available_corr) >= 2:
            corr_matrix = segmented_df[available_corr].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, ax=ax)
            ax.set_title('Specification Correlation Analysis', fontsize=16, fontweight='bold')
            
            if theme_mode == "Dark Mode":
                fig.patch.set_facecolor('#1E2127')
                ax.set_facecolor('#1E2127')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("Not enough data for market segmentation. Please adjust your filters.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px; color: {text_color};'>
    <p><strong>📊 Dataset:</strong> Mobiles Dataset (2025) – Kaggle</p>
    <p><strong>🤖 Technology:</strong> AI-Powered Analysis with Python, Pandas, Streamlit</p>
    <p><strong>🎯 Purpose:</strong> Advanced Data Science & Machine Learning Portfolio Project</p>
    <p><strong>👨‍💻 Developed By:</strong> AI & Data Science Student</p>
    <p><strong>📅 Year:</strong> {datetime.now().year}</p>
</div>
""", unsafe_allow_html=True)

