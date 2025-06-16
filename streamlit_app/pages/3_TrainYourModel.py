import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime

# Modülleri import et
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data, preprocess_data
from utils.model_utils import (
    train_classical_model, train_neural_network, evaluate_model, 
    save_model, plot_confusion_matrix, plot_roc_curve, calculate_feature_importance
)
from utils.visualizations import plot_feature_importance, plot_learning_curve, plot_metrics_comparison

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Model Eğitimi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana başlık
st.title("🧠 Kendi Modelinizi Eğitin")

# Sidebar - Veri Yükleme Seçenekleri
st.sidebar.header("Veri Seçenekleri")

data_source = st.sidebar.radio(
    "Veri Kaynağı",
    ["Örnek Veri Seti", "Kendi Verinizi Yükleyin"]
)

df = None

if data_source == "Örnek Veri Seti":
    # Varsayılan veri setini yükle
    df = load_data()
    st.sidebar.success("Wisconsin Breast Cancer veri seti yüklendi!")
    
else:
    uploaded_file = st.sidebar.file_uploader("CSV veya Excel dosyası yükleyin", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Dosya uzantısını kontrol et
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.sidebar.success(f"{uploaded_file.name} başarıyla yüklendi!")
        except Exception as e:
            st.sidebar.error(f"Hata: {e}")
    else:
        st.sidebar.info("Lütfen bir dosya yükleyin veya örnek veri setini kullanın.")
        # Örnek veri setini yükle
        df = load_data()
        st.sidebar.success("Wisconsin Breast Cancer veri seti yüklendi!")

# Ana sayfa içeriği
if df is not None:
    # Veri bilgilerini göster
    st.subheader("Veri Seti Önizleme")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown(f"**Veri Boyutu:** {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # Eksik değerlerin %40'ından fazlası olan sütunları göster
    missing_percentage = df.isnull().mean()
    problematic_cols = missing_percentage[missing_percentage > 0.4].index.tolist()
    
    if problematic_cols:
        st.warning(f"⚠️ Aşağıdaki sütunlarda %40'tan fazla eksik veri var ve ön işleme sırasında silinecek (hedef sütun hariç):\n{', '.join(problematic_cols)}")
    
    # Veri ön işleme ayarları
    st.header("Veri Ön İşleme Ayarları")
    
    # Hedef sütun seçimi
    # Eğer örnek veri seti kullanılıyorsa varsayılan hedef sütun "target" olacak
    default_target_index = 0  # Varsayılan olarak boş seçenek
    
    if data_source == "Örnek Veri Seti" and "target" in df.columns:
        # Örnek veri seti için "target" sütununu seçili hale getir
        default_target_index = df.columns.tolist().index("target") + 1  # +1 çünkü boş seçenek ekledik
    
    target_column = st.selectbox(
        "Hedef Sütunu Seçin",
        options=[""] + df.columns.tolist(),  # Boş seçenek eklendi
        index=default_target_index  # Örnek veri setiyse target, değilse boş
    )
    
    # Hedef sütun seçilmediğinde uyarı göster
    if not target_column:
        st.warning("⚠️ Lütfen hedef sütunu seçin. Aksi takdirde varsayılan hedef sütun kullanılacak!")
    # Hedef sütun problematik sütunlar arasındaysa uyarı göster
    elif target_column in problematic_cols:
        st.error(f"⚠️ Seçilen hedef sütun '{target_column}' eksik verilerin çok olduğu bir sütun, ancak ön işleme sırasında korunacak.")
    
    # Eğitimden çıkarılacak sütunları seçme
    st.subheader("Eğitimden Çıkarılacak Sütunlar")
    
    # Hedef sütun dışındaki tüm sütunları göster
    available_columns = [col for col in df.columns if col != target_column]
    
    columns_to_drop = st.multiselect(
        "Eğitimden Çıkarılacak Sütunları Seçin",
        options=available_columns,
        default=[],
        help="Eğitim sürecine dahil etmek istemediğiniz sütunları seçin. Bu sütunlar veri setinden çıkarılacaktır."
    )
    
    if columns_to_drop:
        st.info(f"Aşağıdaki sütunlar eğitimden çıkarılacak: {', '.join(columns_to_drop)}")
    
    # Test seti oranı
    test_size = st.slider("Test Seti Oranı", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    # Scaler seçimi
    st.subheader("Veri Ölçeklendirme Ayarları")
    scaler_type = st.selectbox(
        "Veri Ölçeklendirme Yöntemi",
        options=["StandardScaler", "MinMaxScaler", "NoScaler"],
        index=0,
        help="StandardScaler: Veriyi ortalaması 0, standart sapması 1 olacak şekilde ölçeklendirir. MinMaxScaler: Veriyi 0-1 aralığına ölçeklendirir. NoScaler: Herhangi bir ölçeklendirme yapmaz."
    )
    
    # Seçilen scaler hakkında bilgi
    if scaler_type == "StandardScaler":
        st.info("**StandardScaler**: Veriyi standartlaştırır (z-score normalizasyonu). Her özellik için ortalama 0, standart sapma 1 olur.")
    elif scaler_type == "MinMaxScaler":
        st.info("**MinMaxScaler**: Veriyi 0-1 aralığına ölçeklendirir. Minimum değer 0, maksimum değer 1 olur.")
    elif scaler_type == "NoScaler":
        st.warning("**Scaler Yok**: Herhangi bir ölçeklendirme yapılmaz. Veriler orijinal ölçeğinde kalır. Bu durum bazı algoritmaların performansını olumsuz etkileyebilir.")
    
    # Model seçimi
    st.header("Model Seçimi")
    
    model_type = st.selectbox(
        "Model Türü",
        options=["Klasik Makine Öğrenmesi", "Yapay Sinir Ağı"]
    )
    
    # Klasik model türleri
    if model_type == "Klasik Makine Öğrenmesi":
        selected_model = st.selectbox(
            "Algoritma",
            options=["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree", "Gradient Boosting", "MLP"]
        )
        
        # Model parametreleri - her model için özel parametreler
        st.subheader("Model Parametreleri")
        
        if selected_model == "Random Forest":
            n_estimators = st.slider("Ağaç Sayısı", min_value=10, max_value=500, value=100, step=10)
            max_depth = st.slider("Maksimum Derinlik", min_value=2, max_value=30, value=10, step=1)
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42
            }
            model_code = "random_forest"
            
        elif selected_model == "Logistic Regression":
            C = st.select_slider("Düzenlileştirme Parametresi (C)", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            solver = st.selectbox("Çözücü", options=["liblinear", "lbfgs", "newton-cg", "sag"], index=0)
            model_params = {
                "C": C,
                "solver": solver,
                "max_iter": 1000,
                "random_state": 42
            }
            model_code = "logistic_regression"
            
        elif selected_model == "SVM":
            C = st.select_slider("Düzenlileştirme Parametresi (C)", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            kernel = st.selectbox("Çekirdek Fonksiyonu", options=["linear", "rbf", "poly", "sigmoid"], index=1)
            gamma = st.select_slider("Gamma", options=["scale", "auto", 0.001, 0.01, 0.1, 1.0], value="scale")
            model_params = {
                "C": C,
                "kernel": kernel,
                "gamma": gamma,
                "random_state": 42
            }
            model_code = "svm"
            
        elif selected_model == "KNN":
            n_neighbors = st.slider("Komşu Sayısı", min_value=1, max_value=20, value=5, step=1)
            weights = st.selectbox("Ağırlık Fonksiyonu", options=["uniform", "distance"], index=0)
            model_params = {
                "n_neighbors": n_neighbors,
                "weights": weights
            }
            model_code = "knn"
            
        elif selected_model == "Decision Tree":
            max_depth = st.slider("Maksimum Derinlik", min_value=2, max_value=30, value=5, step=1)
            criterion = st.selectbox("Kriter", options=["gini", "entropy"], index=0)
            model_params = {
                "max_depth": max_depth,
                "criterion": criterion,
                "random_state": 42
            }
            model_code = "decision_tree"
            
        elif selected_model == "Gradient Boosting":
            n_estimators = st.slider("Ağaç Sayısı", min_value=10, max_value=500, value=100, step=10)
            learning_rate = st.select_slider("Öğrenme Oranı", options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5], value=0.1)
            max_depth = st.slider("Maksimum Derinlik", min_value=2, max_value=10, value=3, step=1)
            model_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "random_state": 42
            }
            model_code = "gradient_boosting"
            
        elif selected_model == "MLP":
            hidden_layer_sizes = st.text_input("Gizli Katman Boyutları (virgülle ayrılmış)", value="100,50")
            activation = st.selectbox("Aktivasyon Fonksiyonu", options=["relu", "tanh", "logistic"], index=0)
            max_iter = st.slider("Maksimum İterasyon", min_value=100, max_value=1000, value=300, step=100)
            
            # Gizli katman boyutlarını tuple'a dönüştür
            hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
            
            model_params = {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": activation,
                "max_iter": max_iter,
                "random_state": 42
            }
            model_code = "mlp"
            
    # Yapay Sinir Ağı modeli
    else:
        st.subheader("Yapay Sinir Ağı Parametreleri")
        
        # Katman sayısı
        n_layers = st.slider("Gizli Katman Sayısı", min_value=1, max_value=5, value=2, step=1)
        
        # Katman boyutları
        layers = []
        for i in range(n_layers):
            layer_size = st.slider(f"{i+1}. Katman Nöron Sayısı", min_value=8, max_value=256, value=64 // (2**min(i, 2)), step=8)
            layers.append(layer_size)
        
        # Aktivasyon fonksiyonu
        activation = st.selectbox("Aktivasyon Fonksiyonu", options=["relu", "tanh", "sigmoid"], index=0)
        
        # Dropout oranı
        dropout_rate = st.slider("Dropout Oranı", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        
        # Optimizer
        optimizer = st.selectbox("Optimizer", options=["adam", "sgd", "rmsprop"], index=0)
        
        # Batch size ve epoch
        batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
        epochs = st.slider("Epoch Sayısı", min_value=10, max_value=200, value=50, step=10)
        
        model_params = {
            "layers": layers,
            "activation": activation,
            "dropout_rate": dropout_rate,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "epochs": epochs
        }
        
        model_code = "neural_network"
        selected_model = "Neural Network"
    
    # Model adı
    model_name = st.text_input("Model Adı", value=f"{selected_model.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Eğitim butonu
    train_button = st.button("Modeli Eğit", type="primary")
    
    if train_button:
        st.info("Model eğitimi başlatılıyor... Lütfen bekleyin.")
        
        # Veri ön işleme
        try:
            with st.spinner("Veriler ön işleniyor..."):
                # Kullanıcının seçtiği çıkarılacak sütunları preprocessing fonksiyonuna ilet
                X_train, X_test, y_train, y_test, scaler, warnings = preprocess_data(
                    df, 
                    target_column, 
                    test_size,
                    columns_to_drop=columns_to_drop,
                    scaler_type=scaler_type
                )
                
                # Eğitim ve test seti bilgisi
                st.success(f"Veriler ön işlendi. Eğitim seti: {X_train.shape[0]} örnek, Test seti: {X_test.shape[0]} örnek")
                
                # Silinen sütunlar hakkında bilgi
                if columns_to_drop:
                    st.info(f"Seçtiğiniz {len(columns_to_drop)} sütun eğitimden çıkarıldı: {', '.join(columns_to_drop)}")
                
                # Hedef sütun seçilmediğinde veya korunduğunda uyarılar
                if warnings["target_not_selected"]:
                    st.warning(f"⚠️ Hedef sütun seçilmediği için varsayılan sütun '{warnings['target_column_used']}' kullanıldı.")
                
                if warnings["target_protected_from_deletion"]:
                    st.error(f"⚠️ Seçilen hedef sütun '{target_column}' eksik verilerin çok olduğu bir sütun, ancak ön işleme sırasında korundu.")
                
                # Eğitimde kullanılan özellik isimlerini belirle (özellik önemi için gerekli)
                if isinstance(df, pd.DataFrame):
                    all_features = [col for col in df.columns if col != target_column]
                    # Çıkarılan sütunları hariç tut
                    if columns_to_drop:
                        feature_names = [col for col in all_features if col not in columns_to_drop]
                    else:
                        feature_names = all_features
                else:
                    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # Model eğitimi
            with st.spinner(f"{selected_model} modeli eğitiliyor..."):
                start_time = time.time()
                
                if model_code == "neural_network":
                    model, model_info = train_neural_network(X_train, y_train, X_train.shape[1], model_params)
                    model_type_str = "neural_network"
                else:
                    model, model_info = train_classical_model(X_train, y_train, model_code, model_params)
                    model_type_str = "classical"
                
                training_time = time.time() - start_time
                
                st.success(f"Model başarıyla eğitildi. Eğitim süresi: {training_time:.2f} saniye")
            
            # Model değerlendirme
            with st.spinner("Model değerlendiriliyor..."):
                metrics = evaluate_model(model, X_test, y_test, model_type_str)
                
                # Değerlendirme metriklerini göster
                st.subheader("Model Performans Metrikleri")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Doğruluk (Accuracy)", f"{metrics['accuracy']:.4f}")
                col2.metric("Kesinlik (Precision)", f"{metrics['precision']:.4f}")
                col3.metric("Hassasiyet (Recall)", f"{metrics['recall']:.4f}")
                col4.metric("F1 Skoru", f"{metrics['f1_score']:.4f}")
                
                # ROC-AUC değeri
                if metrics['roc_auc'] is not None:
                    st.metric("ROC AUC Skoru", f"{metrics['roc_auc']:.4f}")
                
                # Karmaşıklık matrisi ve ROC eğrisi
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Karmaşıklık Matrisi")
                    cm_fig = plot_confusion_matrix(np.array(metrics['confusion_matrix']))
                    st.pyplot(cm_fig)
                    
                with col2:
                    # ROC eğrisi için tahmin olasılıkları gerekiyor
                    st.subheader("ROC Eğrisi")
                    
                    if model_type_str == "classical":
                        if hasattr(model, "predict_proba"):
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                        else:
                            # SVC gibi modeller için decision_function kullan
                            if hasattr(model, "decision_function"):
                                y_pred_proba = model.decision_function(X_test)
                            else:
                                y_pred_proba = model.predict(X_test)
                    else:  # neural_network
                        y_pred_proba = model.predict(X_test).ravel()
                    
                    roc_fig = plot_roc_curve(y_test, y_pred_proba)
                    st.pyplot(roc_fig)
                
                # Yapay sinir ağı için eğitim geçmişi
                if model_code == "neural_network" and "history" in model_info:
                    st.subheader("Eğitim Geçmişi")
                    history_fig = plot_learning_curve(model_info["history"])
                    st.pyplot(history_fig)
                
                # Özellik önemi (eğer model destekliyorsa)
                if model_code != "neural_network" and model_code not in ["knn"]:
                    st.subheader("Özellik Önemi")
                    
                    # Özellik önemini hesapla (yukarıda hesaplanan feature_names kullan)
                    feature_importance = calculate_feature_importance(model, feature_names, model_type_str)
                    
                    # Özellik önemini görselleştir
                    importance_fig = plot_feature_importance(feature_importance)
                    st.pyplot(importance_fig)
                
                # Preprocessing bilgilerini hazırla
                preprocessing_info = {
                    "columns_to_drop": columns_to_drop or [],
                    "target_column": target_column,
                    "feature_names": feature_names,
                    "test_size": test_size,
                    "scaler_type": scaler_type
                }
                
                # Modeli kaydet
                save_path = save_model(model, model_info, metrics, model_name, directory="models/custom", preprocessing_info=preprocessing_info, scaler=scaler)
                st.success(f"Model başarıyla kaydedildi: {save_path}")
                
                # Model özeti
                st.subheader("Model Özeti")
                
                st.write(f"**Model Adı:** {model_name}")
                st.write(f"**Model Türü:** {selected_model}")
                st.write(f"**Doğruluk (Accuracy):** {metrics['accuracy']:.4f}")
                st.write(f"**Eğitim Süresi:** {training_time:.2f} saniye")
                
                st.info("""
                **Not:** Eğittiğiniz model "models/custom" klasörüne kaydedildi. 
                Modeli "Hazır Modeller" ve "Tahmin" sayfalarında kullanabilirsiniz.
                """)
                
        except Exception as e:
            st.error(f"Model eğitimi sırasında bir hata oluştu: {str(e)}")
            st.exception(e)
            
else:
    st.error("Veri yüklenemedi. Lütfen veri kaynağınızı kontrol edin.")
