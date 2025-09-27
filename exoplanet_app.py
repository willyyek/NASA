import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import numpy as np

# Custom CSS for gradient dark blue header + sidebar
st.markdown(
    """
    <style>
    /* 顶部导航栏（Header） */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #001f3f, #003366);
        color: white;
    }

    /* 左侧 Sidebar 渐变 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #00264d, #004080);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 放在 app 开头
st.markdown(
    """
    <style>
    /* 整体背景黑色 */
    .stApp {
        background-color: #000000;
        color: white !important;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* 标题 (title, header, subheader) 白色 + NASA 蓝 */
    h1, h2, h3, h4, h5, h6 {
        color: #00BFFF !important;  /* NASA 蓝色 */
        font-weight: bold;
    }

    /* 普通文字 */
    p, label, span, div {
        color: white !important;
    }

    /* 按钮设计 */
    .stButton>button {
        background-color: #0B3D91;  /* 深蓝色 */
        color: white;
        border-radius: 10px;
        border: 1px solid #1E90FF;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1E90FF;  /* 浅蓝 hover */
        color: black;
    }

    /* 输入框 (number_input, text_input 等) */
    .stTextInput>div>div>input, .stNumberInput input {
        background-color: #111111;
        color: white !important;
        border: 1px solid #1E90FF;
        border-radius: 5px;
    }

    /* 下拉菜单 */
    .stSelectbox div[data-baseweb="select"]>div {
        background-color: #111111;
        color: white !important;
        border: 1px solid #1E90FF;
        border-radius: 5px;
    }

    /* 滑动条 slider 颜色 */
    .stSlider [role="slider"] {
        background-color: #1E90FF !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* 链接默认样式 */
    a {
        color: #00BFFF !important;       /* 更亮蓝 (DodgerBlue) */
        text-decoration: none !important; /* 去掉下划线 */
        font-weight: bold;
    }
    /* 鼠标悬停样式 */
    a:hover {
        color: #40CFFF !important;       /* 更浅亮蓝 */
        text-decoration: underline !important; /* 悬停时显示下划线 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* 只针对 dataframe 内部的 grid */
    div[data-testid="stDataFrame"] div[role="grid"] {
        background-color: #001F3F !important;  /* 深蓝背景 */
        color: white !important;               /* 白字 */
    }

    /* 表格格子 */
    div[data-testid="stDataFrame"] div[role="gridcell"] {
        background-color: #001F3F !important;
        color: white !important;
        border: 1px solid #00BFFF !important;  /* NASA 蓝边框 */
    }

    /* 列名表头 */
    div[data-testid="stDataFrame"] div[role="columnheader"] {
        background-color: #003366 !important;  /* 稍深的蓝色 */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 加载训练好的模型
model = pickle.load(open("exoplanet_model.pkl", "rb"))

st.set_page_config(page_title="🚀 NASA Exoplanet Classifier", layout="wide")

# --- Sidebar navigation ---
st.sidebar.title("🔭 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Novice Mode", "Researcher Mode"])

# --- Home Page ---
if page == "Home":
    st.title("🚀 NASA Exoplanet Classifier")

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
        #background: rgba(0,0,0,0); /* 顶部透明 */
    }

    [data-testid="stToolbar"] {
        right: 2rem;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.subheader("🌌 Galactic Explorer 117")
    st.markdown(
        """
        Welcome to our Exoplanet Classifier!  
        Choose one of the modes from the sidebar:
        - **Novice Mode** 🟢 : For beginners, explore planets by entering basic parameters. (Default dataset: 🔗 <a href="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative">NASA Kepler Objects of Interest(KOI)</a>)
        - **Researcher Mode** 🔬 : For advanced users, upload datasets, train models, and analyze results.  
        """,
        unsafe_allow_html=True
    )

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
        #background: rgba(0,0,0,0); /* 顶部透明 */
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
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=120)

    st.title("🚀 NASA Exoplanet Classifier")
    st.write("<h3 style='text-align: center; color: white;'>Analyze Kepler exoplanet data and classify candidates into Confirmed, Candidate, or False Positive</h3>", unsafe_allow_html=True)

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
        #background: rgba(0,0,0,0); /* 顶部透明 */
    }

    [data-testid="stToolbar"] {
        right: 2rem;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.write("Here you can upload new datasets, retrain the model, and analyze accuracy.")

    st.markdown(
        """
        <style>
        /* File uploader 外框 */
        [data-testid="stFileUploader"] section {
            background-color: #001f3f;   /* 深蓝色背景 */
            border: 1px solid #00BFFF;   /* 浅蓝色边框 */
            border-radius: 8px;
        }

        /* File uploader 内部文字 */
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] p {
            color: white !important;   /* 白色字体 */
        }

        /* 上传按钮 */
        [data-testid="stFileUploader"] button {
            background-color: #111111;   /* 按钮黑色 */
            color: white !important;     /* 按钮文字白色 */
            border: 1px solid #555555;
            border-radius: 6px;
        }

        [data-testid="stFileUploader"] button:hover {
            background-color: #222222;   /* hover 时稍微亮一点 */
            border: 1px solid #888888;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("📂 Upload dataset", type=["csv", "txt", "tsv", "xlsx"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, comment="#", sep=None, engine="python")
            st.success("✅ File loaded successfully!")
            st.dataframe(data)

            # --- Choose Features & Target ---
            st.subheader("⚙️ Model Training")
            all_columns = data.columns.tolist()
            target_col = st.selectbox("Select Target Column (e.g., koi_disposition)", all_columns)
            feature_cols = st.multiselect("Select Feature Columns", all_columns, default=all_columns)

            # --- Model Selection ---
            model_choice = st.radio("Choose Model", ["RandomForest", "LightGBM"])
            mode = st.radio("Select Training Mode", ["Manual Hyperparameters", "Auto Hyperparameter Tuning"])

            # --- Manual Mode ---
            if mode == "Manual Hyperparameters":
                st.subheader("🎛️ Manual Hyperparameter Tuning")

                if model_choice == "RandomForest":
                    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 200, 50)
                    max_depth = st.slider("Max Depth", 2, 20, 10)
                    min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
                    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

                else:  # LightGBM
                    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 200, 50)
                    max_depth = st.slider("Max Depth", -1, 20, 6)  # -1 = no limit
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)

                if st.button("🚀 Train Model"):
                    if len(feature_cols) > 0:
                        with st.spinner("🛰️ Training model... Please wait while the algorithm orbits the data galaxy 🌌"):
                            import time
                            time.sleep(2)  # 这里可以模拟loading，真实情况是训练时间本身
            
                            X = data[feature_cols].select_dtypes(include=['number']).fillna(0)
                            y = data[target_col]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            if model_choice == "RandomForest":
                                model = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=42
                                )
                            else:
                                import lightgbm as lgb
                                model = lgb.LGBMClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    random_state=42
                                )

                            model.fit(X_train, y_train)

                        # 🚀 出spinner后显示结果
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        st.success(f"✅ {model_choice} trained! Accuracy: **{acc:.2f}**")

                        import joblib
                        joblib.dump(model, "exoplanet_model.pkl")
                        st.info("💾 Model saved as `exoplanet_model.pkl`")

                        st.subheader("📊 Classification Report")
                        st.text(classification_report(y_test, y_pred))

                        st.subheader("🔎 Confusion Matrix")
                        fig, ax = plt.subplots()
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
                        st.pyplot(fig)

            # --- Auto Mode ---
            elif mode == "Auto Hyperparameter Tuning":
                st.subheader("🤖 Auto Hyperparameter Tuning with GridSearchCV")

                if st.button("🔍 Run Grid Search"):
                    if len(feature_cols) > 0:
                        with st.spinner("🚀 Running Grid Search... Exploring hyperparameter galaxies, please hold tight 🌌🛰️"):
                            X = data[feature_cols].select_dtypes(include=['number']).fillna(0)
                            y = data[target_col]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            from sklearn.model_selection import GridSearchCV

                            if model_choice == "RandomForest":
                                param_grid = {
                                    "n_estimators": [100, 200, 300],
                                    "max_depth": [5, 10, 15],
                                    "min_samples_split": [2, 5, 10],
                                    "min_samples_leaf": [1, 2, 4],
                                }
                                grid = GridSearchCV(
                                    RandomForestClassifier(random_state=42),
                                    param_grid,
                                    cv=3,
                                    n_jobs=-1,
                                    verbose=1
                                )

                            else:  # LightGBM
                                import lightgbm as lgb
                                param_grid = {
                                    "n_estimators": [100, 200, 300],
                                    "max_depth": [-1, 6, 12],
                                    "learning_rate": [0.01, 0.05, 0.1],
                                }
                                grid = GridSearchCV(
                                    lgb.LGBMClassifier(random_state=42),
                                    param_grid,
                                    cv=3,
                                    n_jobs=-1,
                                    verbose=1
                                )

                            grid.fit(X_train, y_train)  # 🔄 这里会触发 spinner

                        # 🚀 训练完成后显示结果
                        st.success(f"🎯 Best Parameters: {grid.best_params_}")
                        best_model = grid.best_estimator_

                        y_pred = best_model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        st.success(f"✅ Best {model_choice} Accuracy: **{acc:.2f}**")

                        import joblib
                        joblib.dump(best_model, "exoplanet_model.pkl")
                        st.info("💾 Best model saved as `exoplanet_model.pkl`")

                        st.subheader("📊 Classification Report")
                        st.text(classification_report(y_test, y_pred))

                        st.subheader("🔎 Confusion Matrix")
                        fig, ax = plt.subplots()
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens", ax=ax)
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
    
    st.markdown(
        """
        **Or train using NASA datasets:**

        - <a href="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative" target="_blank">NASA Kepler Objects of Interest (KOI)</a>  
        - <a href="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI" target="_blank">NASA TESS Objects of Interest (TOI)</a>  
        - <a href="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc" target="_blank">NASA K2 Planets and Candidates</a>  

        ⚠️ These datasets need to be downloaded in CSV format and uploaded here again.
        """,
        unsafe_allow_html=True
    )