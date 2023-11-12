import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import os
from train import *

selected = option_menu(None, ["Dashboard Analysis", "Pattern Analysis"], 
    icons=['archive', 'cloud-upload'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if selected =="Dasbord Analysis":
    st.title("Dasbord Analysis")

if selected =="Pattern Analysis":
    st.title("Pattern Analysis")
    eval, y_train, feature_importance= train()
    eval_smote, y_train_resampled, feature_importance_smote = train_smote()
    
    st.write("Informasi Data sebelum dan sesudah SMOTE:")
    visualisasi_label(y_train, y_train_resampled)
    st.write("Perbandingan Model Machine Learning Sebelum SMOTE")
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            st.write(eval)
        with col_2:
            visualisasi_perbadingan(eval)
    plot_feature_importance(feature_importance)
    st.write("Perbandingan Model Machine Learning Sesudah SMOTE")
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            st.write(eval_smote)
        with col_2:
            visualisasi_perbadingan(eval_smote)
    plot_feature_importance(feature_importance_smote)