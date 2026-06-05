import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="智慧輔具預測系統", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 1.5rem !important; }
    @media print {
        @page { size: A4 portrait; margin: 1cm; }
        body { zoom: 0.85; }
        button { display: none !important; }
        div[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
    }
    </style>
""", unsafe_allow_html=True)

PKL_PATH = os.path.join(os.path.dirname(__file__), 'assistive_device_v2.pkl')

ICD_HINT = """
| 代碼 | 意義 |
|------|------|
| 0 | 無此診斷 |
| 1 | 骨骼肌肉系統 |
| 2 | 神經系統 |
| 4 | 循環系統 |
| 9 | 其他 |
| 11 | 損傷/外傷 |
| 12 | 先天異常 |
| 13 | 精神疾患 |
| 19 | 其他特定 |
"""

st.markdown("<h2 style='font-size:22px; font-weight:bold; margin-bottom:5px;'>輔具需求預測系統</h2>", unsafe_allow_html=True)
st.write("輸入個案基本資料與診斷編碼，系統自動預測適合的輔具類別。")
st.markdown("<hr style='margin:10px 0 15px'>", unsafe_allow_html=True)

if not os.path.exists(PKL_PATH):
    st.error(f"找不到模型檔案：{PKL_PATH}")
    st.stop()

pipeline = joblib.load(PKL_PATH)
imputer  = pipeline['imputer']
model    = pipeline['model']
features = pipeline['features']

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("#### 1. 基本生理數值")
    age = st.number_input("年齡", value=75.0, format="%.1f", step=1.0)
    ht  = st.number_input("身高 (cm)", value=160.0, format="%.1f", step=0.5)
    wt  = st.number_input("體重 (kg)", value=60.0, format="%.1f", step=0.5)
    bmi = wt / ((ht / 100) ** 2) if ht > 0 else 0.0
    st.number_input("BMI（自動計算）", value=round(bmi, 1), format="%.1f", disabled=True)

with col2:
    st.markdown("#### 2. 臨床評估與身分指標")
    func_type        = st.number_input("功能型態", value=22.0, format="%.1f", step=1.0)
    disability_level = st.number_input("失能等級", value=0.0, format="%.1f", step=1.0)
    category_of_care = st.number_input("照護類別", value=1.0, format="%.1f", step=1.0)
    master_placement = st.number_input("主要安置", value=1.0, format="%.1f", step=1.0)
    official_rank    = st.number_input("官方等級", value=8.0, format="%.1f", step=1.0)
    address          = st.number_input("居住地代碼", value=1.0, format="%.1f", step=1.0)

with col3:
    st.markdown("#### 3. 診斷系統編碼")
    with st.expander("編碼對照表"):
        st.markdown(ICD_HINT)
    st.caption("無此診斷請填 0")
    icd_main = st.number_input("主診斷", value=0.0, format="%.1f", step=1.0)
    icd_1    = st.number_input("次診斷 1", value=0.0, format="%.1f", step=1.0)
    icd_2    = st.number_input("次診斷 2", value=0.0, format="%.1f", step=1.0)
    icd_3    = st.number_input("次診斷 3", value=0.0, format="%.1f", step=1.0)
    icd_4    = st.number_input("次診斷 4", value=0.0, format="%.1f", step=1.0)

st.markdown("<hr style='margin:15px 0'>", unsafe_allow_html=True)

btn_col, result_col = st.columns([1, 2])

with btn_col:
    predict_btn = st.button("執行輔具需求預測", type="primary", use_container_width=True)

with result_col:
    if predict_btn:
        input_dict = {
            'Category of care':   category_of_care,
            'Disability level':   disability_level,
            'FUNC_TYPE':          func_type,
            'HT_y':               ht,
            'WT_y':               wt,
            'BMI':                bmi,
            'age':                age,
            'address':            address,
            'master placement':   master_placement,
            'official rank':      official_rank,
            'ICD10CM_CODE':       icd_main,
            'has_ICD10CM_CODE':   int(icd_main != 0),
            'ICD10CM_CODE_1':     icd_1,
            'has_ICD10CM_CODE_1': int(icd_1 != 0),
            'ICD10CM_CODE_2':     icd_2,
            'has_ICD10CM_CODE_2': int(icd_2 != 0),
            'ICD10CM_CODE_3':     icd_3,
            'has_ICD10CM_CODE_3': int(icd_3 != 0),
            'ICD10CM_CODE_4':     icd_4,
            'has_ICD10CM_CODE_4': int(icd_4 != 0),
        }

        try:
            input_df      = pd.DataFrame([input_dict])[features]
            input_imputed = imputer.transform(input_df)
            pred          = model.predict(input_imputed)[0]
            proba         = model.predict_proba(input_imputed)[0]

            mapping = {
                0: ("類別1：輕度輔具（單拐、手杖等）",     "normal"),
                1: ("類別2：中度輔具（助行器、腋拐等）",    "warning"),
                2: ("類別3：重度輔具（輪椅、電動代步車等）", "error"),
            }

            label, level = mapping.get(int(pred), ("未知類別", "error"))
            confidence   = proba[int(pred)] * 100

            if level == "normal":
                st.success(f"**預測結果：{label}**　（信心度 {confidence:.1f}%）")
            elif level == "warning":
                st.warning(f"**預測結果：{label}**　（信心度 {confidence:.1f}%）")
            else:
                st.error(f"**預測結果：{label}**　（信心度 {confidence:.1f}%）")

            prob_df = pd.DataFrame({
                "類別": ["輕度", "中度", "重度"],
                "機率": [f"{p*100:.1f}%" for p in proba],
            })
            st.markdown("**各類別機率分佈**")
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"預測錯誤：{e}")
    else:
        st.info("點擊左側按鈕執行預測，結果顯示於此。")
