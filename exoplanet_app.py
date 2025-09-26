import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# 加载训练好的模型
model = pickle.load(open("exoplanet_model.pkl", "rb"))

st.set_page_config(page_title="🚀 NASA Exoplanet Classifier", layout="wide")

# --- Sidebar navigation ---
st.sidebar.title("🔭 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Novice Mode", "Researcher Mode"])

# --- Home Page ---
if page == "Home":
    st.title("🚀 NASA Exoplanet Classifier")
    st.subheader("🌌 Galactic Explorer 117")
    st.write("""
    Welcome to our Exoplanet Classifier!  
    Choose one of the modes from the sidebar:
    - **Novice Mode** 🟢 : For beginners, explore planets by entering basic parameters.  
    - **Researcher Mode** 🔬 : For advanced users, upload datasets, train models, and analyze results.  
    """)

# --- Novice Mode ---
elif page == "Novice Mode":
    st.header("🟢 Novice Mode - Quick Classification")

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
    st.write("<h3 style='text-align: center; color: yellow;'>Analyze Kepler exoplanet data and classify candidates into Confirmed, Candidate, or     False Positive", unsafe_allow_html=True)

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

# --- Researcher Mode ---
elif page == "Researcher Mode":
    st.header("🔬 Researcher Mode - Advanced Tools")
    st.write("Here you can upload new datasets, retrain the model, and analyze accuracy.")

    uploaded_file = st.file_uploader("📂 Upload dataset", type=["csv", "txt", "tsv", "xlsx"])

    # 选择数据来源
    data_option = st.radio(
        "📊 Select dataset option:",
        ["Use Default NASA Data", "Upload My Own Data"]
    )

    if data_option == "Use Default NASA Data":
        st.subheader("🚀 Using default NASA dataset")
        # 这里你可以加载本地的 NASA 数据文件（假设是 CSV）
        try:
            nasa_data = pd.read_csv(r"C:\Users\user\Downloads\kepler.csv")
            st.write(nasa_data.head())
            st.success("✅ NASA dataset loaded successfully!")

            # 这里你也可以直接加上训练步骤
            st.info("📌 Ready for training with NASA dataset.")

        except Exception as e:
            st.error(f"❌ Could not load NASA dataset: {e}")

    elif data_option == "Upload My Own Data":
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file, comment="#", sep=None, engine="python")
                st.success("✅ File loaded successfully!")
                st.dataframe(data.head())

                # --- Choose Features & Target ---
                st.subheader("⚙️ Model Training")
                all_columns = data.columns.tolist()
                target_col = st.selectbox("Select Target Column (e.g., koi_disposition)", all_columns)
                feature_cols = st.multiselect("Select Feature Columns", all_columns, default=all_columns[:5])

                if st.button("🚀 Train Model"):
                    if len(feature_cols) > 0:
                        X = data[feature_cols].select_dtypes(include=['number']).fillna(0)
                        y = data[target_col]

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        model = RandomForestClassifier(n_estimators=200, random_state=42)
                        model.fit(X_train, y_train)

                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        st.success(f"✅ Model trained! Accuracy: **{acc:.2f}**")
                    
                        import joblib
                        joblib.dump(model, "exoplanet_model.pkl")
                        st.info("💾 Model saved as `exoplanet_model.pkl`")

                        # Classification report
                        st.subheader("📊 Classification Report")
                        st.text(classification_report(y_test, y_pred))

                        # Confusion Matrix
                        st.subheader("🔎 Confusion Matrix")
                        fig, ax = plt.subplots()
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
                        st.pyplot(fig)

                    else:
                        st.error("⚠️ Please select at least one feature column.")

            except Exception as e:
                st.error(f"❌ Could not read file: {e}")