import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 加载预训练模型
@st.cache_resource
def load_model():
    return joblib.load('rf.pkl')

model = load_model()

# 烧伤类型映射
burn_type_mapping = {
    0: "正常",
    1: "浅二度烧伤",
    2: "深二度烧伤",
    3: "三度烧伤",
    4: "电击烧伤",
    5: "火焰烧伤"
}

# 页面标题
st.title("烧伤分类预测系统")

# 输入表单
with st.form("input_form"):
    st.header("输入烧伤特征数据")
    
    # 假设有5个特征，这里用示例名称，请根据实际特征修改
    feature1 = st.number_input("特征1", min_value=0.0, max_value=100.0, value=25.0)
    feature2 = st.number_input("特征2", min_value=0.0, max_value=100.0, value=30.0)
    feature3 = st.number_input("特征3", min_value=0.0, max_value=100.0, value=40.0)
    feature4 = st.number_input("特征4", min_value=0.0, max_value=100.0, value=20.0)
    feature5 = st.number_input("特征5", min_value=0.0, max_value=100.0, value=50.0)
    
    submitted = st.form_submit_button("预测")

# 预测和显示结果
if submitted:
    # 创建输入数据DataFrame
    input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5]],
                              columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    
    # 预测
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # 显示结果
    st.subheader("预测结果")
    st.success(f"预测烧伤类型: {burn_type_mapping[prediction]}")
    
    # 显示概率
    st.write("各类别概率:")
    for i, prob in enumerate(probabilities):
        st.write(f"{burn_type_mapping[i]}: {prob:.2%}")
    
    # SHAP解释
    st.subheader("特征重要性分析")
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # 绘制SHAP力图
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(fig)
    
    # 详细SHAP力图
    st.write("详细特征影响:")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, input_data, show=False)
    st.pyplot(fig2)