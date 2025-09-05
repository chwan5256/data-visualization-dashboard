import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

# 페이지 설정
st.set_page_config(
    page_title="Interactive Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀
st.title("🎯 Interactive Data Visualization Dashboard")
st.markdown("### Upload your CSV file and explore your data with advanced visualizations")

# 사이드바 설정
st.sidebar.header("📁 File Upload & Settings")

# 파일 업로드
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file to start visualization"
)

def load_sample_data():
    """샘플 데이터 생성 함수"""
    np.random.seed(42)
    n_samples = 653
    
    # RGB 값과 라벨 생성 (예제 데이터)
    data = {
        'blue': np.random.randint(0, 256, n_samples),
        'green': np.random.randint(0, 256, n_samples),
        'red': np.random.randint(0, 256, n_samples),
        'label': np.random.choice([0, 1, 2, 3], n_samples)
    }
    
    return pd.DataFrame(data)

def analyze_data(df):
    """데이터 기본 분석"""
    st.subheader("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # 데이터 타입 정보
    st.subheader("📊 Data Types & Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        st.dataframe(df.dtypes.to_frame(name='Data Type'))
    
    with col2:
        st.write("**Statistical Summary:**")
        st.dataframe(df.describe())
    
    # 데이터 미리보기
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

def create_visualizations(df):
    """다양한 시각화 생성"""
    
    # 숫자형 컬럼 식별
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("No numeric columns found for visualization!")
        return
    
    # 시각화 옵션
    st.sidebar.subheader("🎨 Visualization Options")
    
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Overview Dashboard", "Scatter Plots", "Distribution Plots", "Correlation Analysis", 
         "3D Visualizations", "Advanced Analytics"]
    )
    
    if viz_type == "Overview Dashboard":
        create_overview_dashboard(df, numeric_cols, categorical_cols)
    elif viz_type == "Scatter Plots":
        create_scatter_plots(df, numeric_cols, categorical_cols)
    elif viz_type == "Distribution Plots":
        create_distribution_plots(df, numeric_cols, categorical_cols)
    elif viz_type == "Correlation Analysis":
        create_correlation_analysis(df, numeric_cols)
    elif viz_type == "3D Visualizations":
        create_3d_visualizations(df, numeric_cols, categorical_cols)
    elif viz_type == "Advanced Analytics":
        create_advanced_analytics(df, numeric_cols, categorical_cols)

def create_overview_dashboard(df, numeric_cols, categorical_cols):
    """종합 대시보드 생성"""
    st.subheader("🎯 Overview Dashboard")
    
    # 컬럼이 RGB + label 구조인 경우 특별 처리
    if set(['red', 'green', 'blue', 'label']).issubset(df.columns):
        create_rgb_dashboard(df)
    else:
        create_general_dashboard(df, numeric_cols, categorical_cols)

def create_rgb_dashboard(df):
    """RGB 데이터 특별 대시보드"""
    st.write("🎨 **RGB Color Data Detected!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RGB 분포
        fig = make_subplots(rows=3, cols=1, 
                          subplot_titles=['Red Distribution', 'Green Distribution', 'Blue Distribution'])
        
        fig.add_trace(go.Histogram(x=df['red'], name='Red', marker_color='red', opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['green'], name='Green', marker_color='green', opacity=0.7), row=2, col=1)
        fig.add_trace(go.Histogram(x=df['blue'], name='Blue', marker_color='blue', opacity=0.7), row=3, col=1)
        
        fig.update_layout(height=600, title="RGB Channel Distributions", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 색상 공간 3D 시각화
        fig = px.scatter_3d(df, x='red', y='green', z='blue', color='label',
                           title="3D RGB Color Space",
                           labels={'red': 'Red Channel', 'green': 'Green Channel', 'blue': 'Blue Channel'})
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # 라벨별 평균 RGB 값
    if 'label' in df.columns:
        st.subheader("📊 Label-wise RGB Analysis")
        label_stats = df.groupby('label')[['red', 'green', 'blue']].agg(['mean', 'std']).round(2)
        st.dataframe(label_stats, use_container_width=True)
        
        # 라벨별 RGB 바 차트
        avg_rgb = df.groupby('label')[['red', 'green', 'blue']].mean()
        fig = px.bar(avg_rgb.reset_index(), x='label', y=['red', 'green', 'blue'],
                    title="Average RGB Values by Label", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

def create_general_dashboard(df, numeric_cols, categorical_cols):
    """일반 데이터 대시보드"""
    col1, col2 = st.columns(2)
    
    with col1:
        if len(numeric_cols) >= 2:
            # 첫 번째 산점도
            x_col = st.selectbox("X-axis", numeric_cols, key="overview_x")
            y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="overview_y")
            color_col = st.selectbox("Color by", [None] + categorical_cols + numeric_cols, key="overview_color")
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 분포 히스토그램
        if numeric_cols:
            hist_col = st.selectbox("Select column for histogram", numeric_cols, key="overview_hist")
            fig = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col}")
            st.plotly_chart(fig, use_container_width=True)

def create_scatter_plots(df, numeric_cols, categorical_cols):
    """산점도 생성"""
    st.subheader("🔵 Interactive Scatter Plots")
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for scatter plots!")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_axis = st.selectbox("X-axis", numeric_cols)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
        color_by = st.selectbox("Color by", [None] + categorical_cols + numeric_cols)
        size_by = st.selectbox("Size by", [None] + numeric_cols)
        
        # 추가 옵션
        show_trendline = st.checkbox("Show Trendline")
        log_x = st.checkbox("Log X-axis")
        log_y = st.checkbox("Log Y-axis")
    
    with col2:
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, size=size_by,
                        title=f"{x_axis} vs {y_axis}",
                        trendline="ols" if show_trendline else None,
                        log_x=log_x, log_y=log_y,
                        hover_data=numeric_cols[:3])  # 상위 3개 컬럼만 hover에 표시
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # 상관계수 표시
        if x_axis != y_axis:
            corr = df[x_axis].corr(df[y_axis])
            st.metric("Correlation Coefficient", f"{corr:.3f}")

def create_distribution_plots(df, numeric_cols, categorical_cols):
    """분포 플롯 생성"""
    st.subheader("📈 Distribution Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Histograms", "Box Plots", "Violin Plots"])
    
    with tab1:
        col = st.selectbox("Select column", numeric_cols, key="dist_hist")
        bins = st.slider("Number of bins", 10, 100, 30)
        
        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # 통계 정보
        st.write(f"**Statistics for {col}:**")
        st.write(df[col].describe())
    
    with tab2:
        if categorical_cols:
            col = st.selectbox("Numeric column", numeric_cols, key="dist_box")
            group_by = st.selectbox("Group by", [None] + categorical_cols, key="dist_box_group")
            
            if group_by:
                fig = px.box(df, x=group_by, y=col, title=f"Box Plot: {col} by {group_by}")
            else:
                fig = px.box(df, y=col, title=f"Box Plot: {col}")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available for grouping.")
    
    with tab3:
        if categorical_cols:
            col = st.selectbox("Numeric column", numeric_cols, key="dist_violin")
            group_by = st.selectbox("Group by", categorical_cols, key="dist_violin_group")
            
            fig = px.violin(df, x=group_by, y=col, title=f"Violin Plot: {col} by {group_by}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available for grouping.")

def create_correlation_analysis(df, numeric_cols):
    """상관관계 분석"""
    st.subheader("🔗 Correlation Analysis")
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for correlation analysis!")
        return
    
    # 상관관계 매트릭스
    corr_matrix = df[numeric_cols].corr()
    
    tab1, tab2 = st.tabs(["Correlation Heatmap", "Pairwise Scatter"])
    
    with tab1:
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix Heatmap",
                       color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        
        # 강한 상관관계 표시
        st.subheader("🔍 Strong Correlations")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.info("No strong correlations (|r| > 0.5) found.")
    
    with tab2:
        if len(numeric_cols) <= 6:  # 너무 많은 컬럼은 제한
            fig = px.scatter_matrix(df[numeric_cols], title="Pairwise Scatter Plot Matrix")
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Too many numeric columns for pairwise scatter. Please select specific columns in other visualization types.")

def create_3d_visualizations(df, numeric_cols, categorical_cols):
    """3D 시각화"""
    st.subheader("🌐 3D Visualizations")
    
    if len(numeric_cols) < 3:
        st.error("Need at least 3 numeric columns for 3D visualization!")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_3d = st.selectbox("X-axis", numeric_cols, key="3d_x")
        y_3d = st.selectbox("Y-axis", numeric_cols, index=1, key="3d_y")
        z_3d = st.selectbox("Z-axis", numeric_cols, index=2, key="3d_z")
        color_3d = st.selectbox("Color by", [None] + categorical_cols + numeric_cols, key="3d_color")
        size_3d = st.selectbox("Size by", [None] + numeric_cols, key="3d_size")
    
    with col2:
        fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d, size=size_3d,
                           title=f"3D Scatter: {x_3d} vs {y_3d} vs {z_3d}")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

def create_advanced_analytics(df, numeric_cols, categorical_cols):
    """고급 분석"""
    st.subheader("🧮 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["PCA Analysis", "Statistical Tests", "Data Profiling"])
    
    with tab1:
        if len(numeric_cols) >= 2:
            st.write("**Principal Component Analysis (PCA)**")
            
            # PCA 수행
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            n_components = min(len(numeric_cols), 3)
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # PCA 결과 시각화
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 설명된 분산 비율
                fig = px.bar(x=[f'PC{i+1}' for i in range(n_components)],
                           y=pca.explained_variance_ratio_,
                           title="Explained Variance Ratio by Component")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # PCA 산점도
                if n_components >= 2:
                    color_col = categorical_cols[0] if categorical_cols else None
                    fig = px.scatter(pca_df, x='PC1', y='PC2', 
                                   color=df[color_col] if color_col else None,
                                   title="PCA: First Two Components")
                    st.plotly_chart(fig, use_container_width=True)
            
            # 컴포넌트 로딩
            st.write("**Component Loadings:**")
            loadings = pd.DataFrame(pca.components_.T, 
                                  columns=[f'PC{i+1}' for i in range(n_components)],
                                  index=numeric_cols)
            st.dataframe(loadings, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for PCA analysis.")
    
    with tab2:
        st.write("**Basic Statistical Information**")
        
        # 정규성 검정 결과 (시각적)
        if numeric_cols:
            col = st.selectbox("Select column for normality check", numeric_cols, key="normality")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Q-Q 플롯
                fig = px.scatter(x=np.sort(np.random.normal(0, 1, len(df))),
                               y=np.sort((df[col] - df[col].mean()) / df[col].std()),
                               title=f"Q-Q Plot for {col}")
                fig.add_scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='Normal line')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 히스토그램 + 정규분포
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("**Comprehensive Data Profiling**")
        
        # 데이터 품질 체크
        quality_metrics = {
            'Total Rows': len(df),
            'Total Columns': len(df.columns),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
            'Memory Usage (MB)': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(quality_df, use_container_width=True)
        
        # 각 컬럼별 상세 정보
        st.write("**Column Details:**")
        column_info = []
        for col in df.columns:
            col_info = {
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Null Count': df[col].isnull().sum(),
                'Null Percentage': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
                'Unique Values': df[col].nunique()
            }
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'Mean': f"{df[col].mean():.2f}",
                    'Std': f"{df[col].std():.2f}",
                    'Min': f"{df[col].min():.2f}",
                    'Max': f"{df[col].max():.2f}"
                })
            column_info.append(col_info)
        
        st.dataframe(pd.DataFrame(column_info), use_container_width=True)

# 메인 앱 로직
if uploaded_file is not None:
    try:
        # 파일 읽기
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # 데이터 분석 및 시각화
        analyze_data(df)
        
        st.divider()
        
        create_visualizations(df)
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please ensure your file is a valid CSV format.")

else:
    # 샘플 데이터 사용 옵션
    st.info("👆 Please upload a CSV file to start visualization")
    
    if st.button("🎲 Use Sample Data (RGB Color Dataset)", type="primary"):
        df = load_sample_data()
        st.success("✅ Sample data loaded! This simulates RGB color data similar to your uploaded file.")
        
        # 샘플 데이터 분석
        analyze_data(df)
        st.divider()
        create_visualizations(df)

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>🚀 Interactive Data Visualization Dashboard | Built with Streamlit & Plotly</p>
    <p>📊 Upload your CSV and explore your data with advanced analytics</p>
</div>
""", unsafe_allow_html=True)
