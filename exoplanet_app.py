import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import time
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


page_bg_video = """
<style>
.stApp {
    background: transparent !important;
}
video#bgvid {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: -1;
    pointer-events: none;
}
</style>

<script>
const existing = document.getElementById("bgvid");
if (!existing) {
    var video = document.createElement("video");
    video.id = "bgvid";
    video.autoplay = true;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.innerHTML = '<source src="https://github.com/chengkiet2020-byte/exoplanet-app/raw/refs/heads/main/videos/earthroute%20(1)%20(1).mp4" type="video/mp4">';
    document.body.appendChild(video);
}
</script>
"""
st.markdown(page_bg_video, unsafe_allow_html=True)


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


st.markdown("""
<style>
/* ============== 强制覆盖下拉菜单（展开的选项列表） ============== */

/* 常见的 BaseWeb / popover 容器和 menu 节点 */
div[data-baseweb="menu"], div[data-baseweb="menu"] * ,
div[class*="baseui-popover"], div[class*="baseui-popover"] * ,
div[class*="baseweb-popover"], div[class*="baseweb-popover"] * {
    background: #000000 !important;
    color: white !important;
}

/* listbox 可能是 ul 或 div，及其所有子项 */
ul[role="listbox"], ul[role="listbox"] * ,
div[role="listbox"], div[role="listbox"] * {
    background: #000000 !important;
    color: white !important;
    border: 1px solid #1E90FF !important;  /* 外框轻微蓝色 */
}

/* 每个选项（有时是 div[role="option"] 或 li[role="option"]） */
div[role="option"], div[role="option"] * ,
li[role="option"], li[role="option"] * {
    background: #000000 !important;
    color: white !important;
    padding: 6px 10px !important;
}

/* Hover 高亮（选项被悬停时） */
div[role="option"]:hover, li[role="option"]:hover,
ul[role="listbox"] li:hover {
    background: #1E90FF !important;  /* NASA 蓝高亮 */
    color: black !important;
}

/* 一些实现使用 rc-virtual-list（虚拟滚动列表），也覆盖它 */
.rc-virtual-list, .rc-virtual-list-holder, .rc-virtual-list-holder-inner,
.rc-virtual-list .rc-virtual-list-scrollbar {
    background: #000000 !important;
    color: white !important;
}

/* 美化滚动条（webkit 浏览器）*/
ul[role="listbox"]::-webkit-scrollbar, div[role="listbox"]::-webkit-scrollbar {
    width: 10px;
}
ul[role="listbox"]::-webkit-scrollbar-track, div[role="listbox"]::-webkit-scrollbar-track {
    background: #000000 !important;
}
ul[role="listbox"]::-webkit-scrollbar-thumb, div[role="listbox"]::-webkit-scrollbar-thumb {
    background: #1E90FF !important;
    border-radius: 6px;
    border: 2px solid #000 !important;
}

/* 限制高度并启用滚动（避免展开过长） */
ul[role="listbox"], div[role="listbox"] {
    max-height: 260px !important;
    overflow-y: auto !important;
}

/* 如果仍然被某些内联类覆盖，可以用 very-specific selector（提高权重） */
div[data-baseweb="menu"] div[role="listbox"] div[role="option"] {
    background: #000000 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Title (st.title 或 markdown #) */
h1 {
    font-size: 100px !important;
}

/* Header (st.header 或 markdown ##) */
h2 {
    font-size: 60px !important;
}

/* Subheader (st.subheader 或 markdown ###) */
h3 {
    font-size: 36px !important;
}

/* 普通 markdown 文本（p, span, div 内文字） */
p, div, span, label {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)


# 放在 app 开头
st.markdown(
    """
    <style>
    /* 整体背景黑色 */
    .stApp {
        background-color: transparent !important;
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
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    [data-testid="stHeader"] {
        background: transparent;
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

st.markdown("""
    <style>
    /* 让 Streamlit spinner 圈圈变成白色 */
    div[data-testid="stSpinner"] div[role="status"]::before {
        border: 0.25em solid rgba(255, 255, 255, 0.2) !important;  /* 半透明外圈 */
        border-top: 0.25em solid white !important;                 /* 白色高亮圈 */
    }

    /* Spinner 的文字颜色 */
    div[data-testid="stSpinner"] div[role="status"] p {
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Streamlit 默认 progressbar 的蓝色 NASA 渐变 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #001f3f, #003366, #1E90FF);
    }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* --- Selectbox & Multiselect 黑底白字 --- */
div[data-baseweb="select"] > div {
    background-color: #111111 !important;   /* 输入框黑色背景 */
    color: white !important;                /* 文字白色 */
    border: 1px solid #1E90FF !important;   /* NASA 蓝边框 */
    border-radius: 5px !important;
}

/* 输入框里的文字 */
div[data-baseweb="select"] span {
    color: white !important;
}

/* 下拉展开的菜单 */
ul[role="listbox"] {
    background-color: #111111 !important;   /* 下拉黑底 */
    color: white !important;
    border: 1px solid #1E90FF !important;
}
ul[role="listbox"] li {
    color: white !important;                /* 下拉选项文字 */
}
ul[role="listbox"] li:hover {
    background-color: #1E90FF !important;   /* Hover 高亮蓝 */
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 下拉展开菜单整体 */
ul[role="listbox"] {
    background-color: #000000 !important;   /* 黑色背景 */
    color: white !important;                /* 白字 */
    border: 1px solid #1E90FF !important;   /* NASA 蓝边框 */
    max-height: 200px;                      /* 限制高度，启用滚动条 */
    overflow-y: auto !important;            /* 确保可以滚动 */
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* 顶部 header 去掉 */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* main 容器去掉所有 padding/margin */
    main[data-testid="stAppViewContainer"] {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* block-container 顶部留 1rem */
    .block-container {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }

    /* AppViewBlockContainer 也统一 */
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 加载训练好的模型
model = pickle.load(open("exoplanet_model.pkl", "rb"))

st.set_page_config(page_title="🚀 NASA Exoplanet Classifier", layout="wide")

# --- Sidebar navigation ---
st.sidebar.subheader("🔭 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Novice Mode", "Researcher Mode"])

# --- Home Page ---
if page == "Home":
    # logo 图标
    col_empty, col_left, col_mid, col_right, col_empty2 = st.columns([0.5, 1, 2, 1, 0.5])

    with col_left:
        st.markdown("<div style='margin-top:-20px'></div>", unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=300)  
    
    with col_right: 
        st.markdown("<div style='margin-left:100px'></div>", unsafe_allow_html=True)  # 加空白
        st.image("https://i.imgur.com/TL2RXUL.png", width=300)

    st.title("Exoscan117 🚀")

    page_bg_video = """
    <style>
    .stApp {
        background: transparent !important; /* 去掉默认背景 */
    }
    video#bgvid {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
        pointer-events: none;  /* 禁止鼠标事件，避免误点 */
    }
    video::-webkit-media-controls {
        display: none !important;  /* 隐藏控制栏 (Chrome, Edge, Safari) */
    }
    video::-moz-media-controls {
        display: none !important;  /* 隐藏控制栏 (Firefox) */
    }
    </style>

    <video autoplay muted loop playsinline id="bgvid">
        <source src="https://github.com/chengkiet2020-byte/exoplanet-app/raw/refs/heads/main/videos/earthroute%20(1)%20(1).mp4" type="video/mp4">
    </video>
    """
    st.markdown(page_bg_video, unsafe_allow_html=True)

    st.subheader("🌌 Galactic Explorer 117")
    st.markdown(
        """
        Welcome to our Exoplanet Classifier!  
        Choose one of the modes from the sidebar:
        - **Novice Mode** 🟢 : For beginners, explore planets by entering basic parameters. (Default dataset: 🔗 <a href="https://exoplanetarchive.ipac.caltech.edu/cgi-    bin/TblView/nph-tblView?app=ExoTbls&config=cumulative">NASA Kepler Objects of Interest(KOI)</a>)
        - **Researcher Mode** 🔬 : For advanced users, upload datasets, train models, and analyze results.  
        """,
        unsafe_allow_html=True
    )

# --- Novice Mode ---
elif page == "Novice Mode":
    st.header("🟢 Novice Mode - Quick Classification")

    # 设置星空背景
    page_bg_video = """
    <style>
    .stApp {
        background: transparent !important; /* 去掉默认背景 */
    }
    video#bgvid {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
        pointer-events: none;  /* 禁止鼠标事件，避免误点 */
    }
    video::-webkit-media-controls {
        display: none !important;  /* 隐藏控制栏 (Chrome, Edge, Safari) */
    }
    video::-moz-media-controls {
        display: none !important;  /* 隐藏控制栏 (Firefox) */
    }
    </style>

    <video autoplay muted loop playsinline id="bgvid">
        <source src="https://github.com/chengkiet2020-byte/exoplanet-app/raw/refs/heads/main/videos/earthroute%20(1)%20(1).mp4" type="video/mp4">
    </video>
    """
    st.markdown(page_bg_video, unsafe_allow_html=True)



    # NASA Logo + 标题
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=120)

    st.title("🚀 NASA Exoplanet Classifier")
    st.write("<h4 style='text-align: center; color: white;'>Analyze Kepler exoplanet data and classify candidates into Confirmed, Candidate, or False Positive</h4>", unsafe_allow_html=True)

    # 页面分为两列
    col_input, col_nothing, col_result = st.columns([3, 1, 3])  # 左边宽一些，右边窄一些

    with col_input:
        st.subheader("🔢 Input Parameters")
        koi_period = st.number_input("Enter Orbital Period (days)", min_value=0.0, step=0.1, value=10.0)
        koi_prad = st.number_input("Enter Planetary Radius (Earth radii)", min_value=0.0, step=0.1, value=1.0)
        koi_duration = st.number_input("Enter Transit Duration (hours)", min_value=0.0, step=0.1, value=5.0)
        koi_depth = st.number_input("Enter Transit Depth (ppm)", min_value=0.0, step=0.1, value=100.0)
        koi_steff = st.number_input("Enter Stellar Effective Temperature (K)", min_value=0.0, step=100.0, value=5500.0)
        koi_srad = st.number_input("Enter Stellar Radius (Solar radii)", min_value=0.0, step=0.1, value=1.0)
        koi_smass = st.number_input("Enter Stellar Mass (Solar masses)", min_value=0.0, step=0.1, value=1.0)

    with col_result:
        st.subheader("🚀 Prediction Result")
        if st.button("🔍 Predict Exoplanet"):
            features = np.array([[koi_period, koi_prad, koi_duration, koi_depth, koi_steff, koi_srad, koi_smass]])
            prediction = model.predict(features)[0]

            nasa_logo_url = "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg"

            if prediction == "CONFIRMED":
                st.image(nasa_logo_url, width=80)
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
    page_bg_video = """
    <style>
    .stApp {
        background: transparent !important; /* 去掉默认背景 */
    }
    video#bgvid {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
        pointer-events: none;  /* 禁止鼠标事件，避免误点 */
    }
    video::-webkit-media-controls {
        display: none !important;  /* 隐藏控制栏 (Chrome, Edge, Safari) */
    }
    video::-moz-media-controls {
        display: none !important;  /* 隐藏控制栏 (Firefox) */
    }
    </style>

    <video autoplay muted loop playsinline id="bgvid">
        <source src="https://github.com/chengkiet2020-byte/exoplanet-app/raw/refs/heads/main/videos/earthroute%20(1)%20(1).mp4" type="video/mp4">
    </video>
    """
    st.markdown(page_bg_video, unsafe_allow_html=True)


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

                        # 🌌 CSS 动态星空 + 星云 Nebula + 火箭
                        st.markdown("""
                            <style>
                            body {
                                background: black;
                                color: white;
                                overflow: hidden;
                            }

                            /* 🌊 流光进度条 */
                            @keyframes shimmer {
                                0% { background-position: -200px 0; }
                                100% { background-position: 200px 0; }
                            }
                            .stProgress > div > div > div > div {
                                background: linear-gradient(90deg, #001f3f, #003366, #1E90FF, #00BFFF);
                                background-size: 400% 100%;
                                animation: shimmer 2s infinite linear;
                            }
                            </style>

                        """, unsafe_allow_html=True)

                        # 🚀 火箭动画 + Spinner
                        rocket_html = """
                        <style>
                        .rocket {
                            position: relative;
                            width: 60px;
                            height: 120px;
                            margin: 50px auto;
                            background: silver;
                            border-radius: 30px;
                        }
                        .rocket::before {
                            content: '';
                            position: absolute;
                            top: -30px;
                            left: 10px;
                            width: 40px;
                            height: 40px;
                            background: gray;
                            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
                        }
                        .flame {
                            position: absolute;
                            bottom: -40px;
                            left: 50%;
                            transform: translateX(-50%);
                            width: 30px;
                            height: 60px;
                            border-radius: 50%;
                            animation: flameAnim 1s infinite alternate;
                        }
                        @keyframes flameAnim {
                            0%   { background: radial-gradient(circle at 50% 0%, orange, red, transparent); }
                            25%  { background: radial-gradient(circle at 50% 0%, yellow, orange, transparent); }
                            50%  { background: radial-gradient(circle at 50% 0%, deepskyblue, dodgerblue, transparent); }
                            75%  { background: radial-gradient(circle at 50% 0%, violet, purple, transparent); }
                            100% { background: radial-gradient(circle at 50% 0%, orange, red, transparent); }
                        }
                        </style>
                        <div class="rocket"><div class="flame"></div></div>
                        """

                        st.markdown(rocket_html, unsafe_allow_html=True)

                        with st.spinner("🛰️ Training model... Please wait while the rocket explores the data galaxy 🌌"):
                            import time
                            time.sleep(2)  # 模拟loading，真实情况会用训练时间
            
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
                        joblib.dump(model, "exoplanet_model_fake.pkl")
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

                if st.button("🚀 Train Model"):
                    if len(feature_cols) > 0:

                        # 🌌 CSS 动态星空 + 星云 Nebula + 火箭
                        st.markdown("""
                            <style>
                            body {
                                background: black;
                                color: white;
                                overflow: hidden;
                            }

                            /* 🌊 流光进度条 */
                            @keyframes shimmer {
                                0% { background-position: -200px 0; }
                                100% { background-position: 200px 0; }
                            }
                            .stProgress > div > div > div > div {
                                background: linear-gradient(90deg, #001f3f, #003366, #1E90FF, #00BFFF);
                                background-size: 400% 100%;
                                animation: shimmer 2s infinite linear;
                            }
                            </style>

                        """, unsafe_allow_html=True)

                        # 🚀 火箭动画 + Spinner
                        rocket_html = """
                        <style>
                        .rocket {
                            position: relative;
                            width: 60px;
                            height: 120px;
                            margin: 50px auto;
                            background: silver;
                            border-radius: 30px;
                        }
                        .rocket::before {
                            content: '';
                            position: absolute;
                            top: -30px;
                            left: 10px;
                            width: 40px;
                            height: 40px;
                            background: gray;
                            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
                        }
                        .flame {
                            position: absolute;
                            bottom: -40px;
                            left: 50%;
                            transform: translateX(-50%);
                            width: 30px;
                            height: 60px;
                            border-radius: 50%;
                            animation: flameAnim 1s infinite alternate;
                        }
                        @keyframes flameAnim {
                            0%   { background: radial-gradient(circle at 50% 0%, orange, red, transparent); }
                            25%  { background: radial-gradient(circle at 50% 0%, yellow, orange, transparent); }
                            50%  { background: radial-gradient(circle at 50% 0%, deepskyblue, dodgerblue, transparent); }
                            75%  { background: radial-gradient(circle at 50% 0%, violet, purple, transparent); }
                            100% { background: radial-gradient(circle at 50% 0%, orange, red, transparent); }
                        }
                        </style>
                        <div class="rocket"><div class="flame"></div></div>
                        """

                        st.markdown(rocket_html, unsafe_allow_html=True)

                        with st.spinner("🛰️ Training model... Please wait while the rocket explores the data galaxy 🌌"):
                            import time
                            time.sleep(2)  # 模拟loading，真实情况会用训练时间

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





















