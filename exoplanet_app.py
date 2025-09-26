import pandas as pd
import streamlit as st
import pickle
import numpy as np

# 加载训练好的模型
model = pickle.load(open("exoplanet_model.pkl", "rb"))

# 设置星空背景
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.nasa.gov/wp-content/uploads/2023/07/asteroid-belt.jpg?resize=2000,1125");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0); /* 顶部透明 */
}

[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# NASA Logo + 标题
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=320)

with col3:
    st.image("https://github.com/chengkiet2020-byte/exoplanet-app/blob/main/logo.png?raw=true", width=410)

st.title("🚀 NASA Exoplanet Classifier")
st.markdown("<h2 style='text-align: center; color: #1E90FF;'>Galactic Explorer 117</h2>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center; color: yellow;'>Analyze Kepler exoplanet data and classify candidates into Confirmed, Candidate, or False Positive", unsafe_allow_html=True)

# 用户输入
koi_period = st.number_input("Enter Orbital Period (days)", min_value=0.0, step=0.1, value=10.0)
koi_prad = st.number_input("Enter Planetary Radius (Earth radii)", min_value=0.0, step=0.1, value=1.0)
koi_duration = st.number_input("Enter Transit Duration (hours)", min_value=0.0, step=0.1, value=5.0)
koi_depth = st.number_input("Enter Transit Depth (ppm)", min_value=0.0, step=0.1, value=100.0)
koi_steff = st.number_input("Enter Stellar Effective Temperature (K)", min_value=0.0, step=100.0, value=5500.0)
koi_srad = st.number_input("Enter Stellar Radius (Solar radii)", min_value=0.0, step=0.1, value=1.0)
koi_smass = st.number_input("Enter Stellar Mass (Solar masses)", min_value=0.0, step=0.1, value=1.0)

# 预测按钮
if st.button("🔍 Predict Exoplanet"):
    features = np.array([[koi_period, koi_prad, koi_duration, koi_depth, koi_steff, koi_srad, koi_smass]])
    prediction = model.predict(features)[0]

    nasa_logo_url = "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg"

    if prediction == "CONFIRMED":
        st.image(nasa_logo_url, width=80)  # 显示NASA logo
        st.success("✅ This is a **Confirmed Exoplanet**!")
        st.markdown("""
        **Explanation:**  
        - A confirmed exoplanet has been validated by astronomers using multiple methods.  
        - It is officially recognized in NASA’s confirmed planet catalog.  
        """)

    elif prediction == "CANDIDATE":
        st.image(nasa_logo_url, width=80)
        st.warning("🟡 This is a **Planet Candidate**, further validation required.")
        st.markdown("""
        **Explanation:**  
        - A planet candidate shows signs of being a planet but has not yet been fully confirmed.  
        - Further observation or analysis is required to rule out false signals.  
        """)

    else:
        st.image(nasa_logo_url, width=80)
        st.error("❌ This is a **False Positive**.")
        st.markdown("""
        **Explanation:**  
        - A false positive means the signal is not caused by a planet.  
        - It could be due to noise, binary stars, or stellar activity instead of an exoplanet.  
        """)

uploaded_file = st.file_uploader("Upload NASA dataset", type=["csv", "txt", "tsv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
            data = pd.read_excel(uploaded_file)
        else:
            try:
                # 尝试逗号分隔 + 忽略注释
                data = pd.read_csv(uploaded_file, sep=",", comment="#")
            except Exception:
                # 如果失败，尝试 Tab 分隔
                data = pd.read_csv(uploaded_file, sep="\t", comment="#")

        st.success("✅ File loaded successfully!")
        st.dataframe(data.head())

    except Exception as e:
        st.error(f"❌ Could not read file: {e}")