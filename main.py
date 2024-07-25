import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans

# 讀取資料
test = pd.read_csv("https://letranger.github.io/working/scores.csv")

# Streamlit 應用程式標題
st.title("分數分佈")

# 設定側邊欄選單
st.sidebar.write('選單')

# 科目選擇下拉選單
option = st.sidebar.selectbox(
    '請選擇科目',
    ['國文', '英文', '數學', '物理']
)

# 顯示資料框
st.write("### 資料", test)

# 將分數依每10分分一組
test['binned'] = pd.cut(test[option], bins=range(0, 101, 10))

# 計算每個區間的頻率
frequency = test['binned'].value_counts().sort_index()
# 轉換為資料框格式，方便 Streamlit 顯示
frequency_df = pd.DataFrame({'分數區間': frequency.index.astype(str), '頻率': frequency.values})

# 使用 Streamlit 顯示長條圖
st.bar_chart(frequency_df.set_index('分數區間'))

# KMeans 聚類
# 使用滑桿選擇 k 值（聚類數量）
k = st.slider("選擇聚類的數量 k", 2, 10, 3)
kmeans = KMeans(n_clusters=k)  # 預設分為 k 群
kmeans.fit(test[['國文', '英文']])  # 使用 '國文' 和 '英文' 進行聚類
clusters = kmeans.predict(test[['國文', '英文']])

# 將聚類結果添加到資料框
test['聚類'] = clusters

# 使用 Plotly 繪製聚類結果的散點圖
fig2 = px.scatter(test, x='國文', y='英文', color='聚類', title='國文與英文分數的聚類結果')
st.plotly_chart(fig2)
