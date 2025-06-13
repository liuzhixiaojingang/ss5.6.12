import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 加载预训练模型
@st.cache_resource
def load_model():
    model = joblib.load('rf.pkl')
    # 添加特征名称映射（关键修改）
    model.feature_names_in_ = ['BG1', 'BG2', 'BG4', 'Cyclic AMP', 'IL-1β']
    return model

model = load_model()

# 烧伤类型映射（保持不变）
burn_type_mapping = {
    'C': "正常",
    'M1': "浅二度烧伤",
    'M2': "深二度烧伤",
    'M3': "三度烧伤",
    'M4': "电击烧伤",
    'M5': "火焰烧伤"
}

# 页面标题（保持不变）
st.title("烧伤分类预测系统")

# 输入表单（保持界面不变）
with st.form("input_form"):
    st.header("输入烧伤特征数据")
    
    # 保持原有变量名不变
    feature1 = st.number_input("BG1", min_value=-8.0, max_value=20.0, value=15.0)
    feature2 = st.number_input("BG2", min_value=-1.000, max_value=2.000, value=0.123)
    feature3 = st.number_input("BG4", min_value=-1.000, max_value=1.000, value=0.145)
    feature4 = st.number_input("Cyclic AMP", min_value=2000, max_value=50000, value=5000)
    feature5 = st.number_input("IL-1β", min_value=200.0, max_value=500.0, value=300.0)

    submitted = st.form_submit_button("预测")

# 预测和显示结果（关键修改在数据转换部分）
if submitted:
    input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5]],
                            columns=['BG1', 'BG2', 'BG4', 'Cyclic AMP', 'IL-1β'])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.subheader("预测结果")
    st.success(f"预测烧伤类型: {burn_type_mapping[prediction]}")
    
    # 修复后的概率显示（方法2）
    st.write("各类别概率:")
    labels = ['C', 'M1', 'M2', 'M3', 'M4', 'M5']
    for label, prob in zip(labels, probabilities):
        st.write(f"{burn_type_mapping[label]}: {prob:.2%}")
    
    