import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Modülleri import et
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data
from utils.model_utils import list_available_models, load_model, apply_model_preprocessing
from utils.visualizations import plot_feature_importance

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Tahmin",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana başlık
st.title("🔮 Meme Kanseri Teşhis Tahmini")

# Tahmin için veri yükleme fonksiyonu
def get_prediction_data(model_feature_names=None, columns_to_drop=None, scaler=None):
    """Tahmin için veri yükleme fonksiyonu"""
    
    data_source = st.radio(
        "Veri Kaynağı",
        ["Örnek Veri", "Elle Giriş", "CSV Dosyası Yükle"]
    )
    
    if data_source == "Örnek Veri":
        df = load_data()
        
        # Örnek veri için bir satır seç
        selected_example = st.slider("Örnek Satır Seçin", min_value=0, max_value=len(df)-1, value=0)
        
        # Hedef sütunu ayır
        X = df.drop('target', axis=1) if 'target' in df.columns else df
        y = df['target'] if 'target' in df.columns else None
        
        # Eğitim sırasında çıkarılan sütunları burada da çıkar
        if columns_to_drop and len(columns_to_drop) > 0:
            available_columns = list(X.columns)
            columns_to_drop_available = [col for col in columns_to_drop if col in available_columns]
            if columns_to_drop_available:
                X = X.drop(columns=columns_to_drop_available)
                st.info(f"Model eğitimi sırasında çıkarılan sütunlar örnek veriden de çıkarıldı: {', '.join(columns_to_drop_available)}")
        
        # Seçilen satırı al
        example_data = X.iloc[selected_example:selected_example+1].copy()
        true_label = y.iloc[selected_example] if y is not None else None
        
        return example_data, true_label, list(X.columns)
    
    elif data_source == "Elle Giriş":
        st.info("Lütfen aşağıdaki özelliklerin değerlerini girin:")
        
        # Model özellik isimlerini kullan (eğer varsa), yoksa örnek veriden al
        if model_feature_names and len(model_feature_names) > 0:
            feature_names = model_feature_names
            # Örnek veri istatistikleri için
            df = load_data()
            X = df.drop('target', axis=1) if 'target' in df.columns else df
            # Eğitim sırasında çıkarılan sütunları burada da çıkar
            if columns_to_drop and len(columns_to_drop) > 0:
                available_columns = list(X.columns)
                columns_to_drop_available = [col for col in columns_to_drop if col in available_columns]
                if columns_to_drop_available:
                    X = X.drop(columns=columns_to_drop_available)
        else:
            # Örnek veriyi yükle ve özellik isimlerini al
            df = load_data()
            X = df.drop('target', axis=1) if 'target' in df.columns else df
            feature_names = list(X.columns)
        
        # Sadece mevcut özelliklerin istatistiklerini hesapla
        available_features = [f for f in feature_names if f in X.columns]
        if len(available_features) != len(feature_names):
            st.warning(f"Bazı özellikler mevcut değil. Sadece {len(available_features)}/{len(feature_names)} özellik kullanılabilir.")
            feature_names = available_features
        
        # Özellikler için örnek istatistikleri hesapla
        feature_stats = X[feature_names].describe().T[['mean', 'min', 'max']]
        
        # Kullanıcı girişi için 3 sütunlu düzen
        col1, col2, col3 = st.columns(3)
        
        user_input = {}
        for i, feature in enumerate(feature_names):
            # Hangi sütunda gösterileceğini belirle
            col_idx = i % 3
            col = [col1, col2, col3][col_idx]
            
            # Min, max ve ortalama değerleri al
            feat_min = float(feature_stats.loc[feature, 'min'])
            feat_max = float(feature_stats.loc[feature, 'max'])
            feat_mean = float(feature_stats.loc[feature, 'mean'])
            
            # Kullanıcıdan değer iste
            with col:
                user_input[feature] = st.number_input(
                    f"{feature}",
                    min_value=float(feat_min * 0.5),
                    max_value=float(feat_max * 1.5),
                    value=float(feat_mean),
                    step=float((feat_max - feat_min) / 100),
                    format="%.4f"
                )
        
        # Kullanıcı girişlerini DataFrame'e dönüştür
        example_data = pd.DataFrame([user_input])
        return example_data, None, feature_names
    
    else:  # CSV Dosyası Yükle
        uploaded_file = st.file_uploader("CSV dosyası yükleyin", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"{uploaded_file.name} başarıyla yüklendi!")
                
                # Veriyi göster
                st.dataframe(df.head(), use_container_width=True)
                
                # Eğer birden fazla satır varsa kullanıcıya hangi satırın tahmin edileceğini seç
                if len(df) > 1:
                    selected_row = st.slider("Tahmin Edilecek Satırı Seçin", min_value=0, max_value=len(df)-1, value=0)
                    example_data = df.iloc[selected_row:selected_row+1].copy()
                else:
                    example_data = df.copy()
                
                # Eğer 'target' sütunu varsa, onu ayır
                if 'target' in example_data.columns:
                    true_label = example_data['target'].iloc[0]
                    example_data = example_data.drop('target', axis=1)
                else:
                    true_label = None
                
                # Eğitim sırasında çıkarılan sütunları burada da çıkar
                if columns_to_drop and len(columns_to_drop) > 0:
                    available_columns = list(example_data.columns)
                    columns_to_drop_available = [col for col in columns_to_drop if col in available_columns]
                    if columns_to_drop_available:
                        example_data = example_data.drop(columns=columns_to_drop_available)
                        st.info(f"Model eğitimi sırasında çıkarılan sütunlar CSV verisinden de çıkarıldı: {', '.join(columns_to_drop_available)}")
                
                return example_data, true_label, list(example_data.columns)
                
            except Exception as e:
                st.error(f"Dosya yüklenirken hata oluştu: {str(e)}")
                return None, None, None
        else:
            st.info("Lütfen bir CSV dosyası yükleyin.")
            return None, None, None

# Modelleri yükle
try:
    available_models = list_available_models()
    
    # Tüm modelleri birleştir
    all_models = available_models["ready"] + available_models["custom"]
    
    # Eğer model yoksa uyarı ver
    if len(all_models) == 0:
        st.warning("""
        Şu an için hiç model bulunmuyor. 
        'Model Eğitimi' sayfasını kullanarak model eğitebilirsiniz veya 'Hazır Modeller' sayfasından demo modelleri deneyebilirsiniz.
        """)
        st.stop()
    
    # Sidebar - Model Seçimi
    st.sidebar.header("Tahmin Ayarları")
    
    # Model seçimi
    selected_model_name = st.sidebar.selectbox(
        "Tahmin için Model Seçin",
        options=[model["name"] for model in all_models],
        index=0
    )
    
    # Seçilen modeli bul
    selected_model_info = next((model for model in all_models if model["name"] == selected_model_name), None)
    
    if selected_model_info:
        # Modelin kategorisini belirle
        model_category = "ready" if selected_model_info in available_models["ready"] else "custom"
        
        # Demo modeller için özel durum
        is_demo = "demo" in selected_model_name
        
        # Model bilgilerini göster
        st.sidebar.subheader("Seçilen Model Bilgileri")
        st.sidebar.write(f"**Model Adı:** {selected_model_info['name']}")
        st.sidebar.write(f"**Model Türü:** {selected_model_info['type']}")
        st.sidebar.write(f"**Doğruluk (Accuracy):** {selected_model_info['accuracy']:.4f}")
        
        # Tahmin eşiği ayarı (binary sınıflandırma için)
        threshold = st.sidebar.slider(
            "Tahmin Eşiği",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Pozitif sınıf (malignant) için tahmin eşiği."
        )
        
        # Model preprocessing bilgilerini al (eğer custom model ise)
        model_feature_names = None
        columns_to_drop = []
        has_preprocessing_info = False
        
        if model_category == "custom":
            try:
                # Sadece preprocessing bilgilerini al
                _, _, _, preprocessing_info, _ = load_model(selected_model_name, directory=f"models/{model_category}")
                model_feature_names = preprocessing_info.get("feature_names", None)
                columns_to_drop = preprocessing_info.get("columns_to_drop", [])
                scaler_type = preprocessing_info.get("scaler_type", "StandardScaler")
                
                # Eğer preprocessing bilgisi varsa
                if preprocessing_info and (model_feature_names or columns_to_drop):
                    has_preprocessing_info = True
                    if columns_to_drop:
                        st.sidebar.info(f"Bu model eğitilirken şu sütunlar çıkarılmıştı: {', '.join(columns_to_drop)}")
                    st.sidebar.info(f"Bu model eğitilirken kullanılan scaler: {scaler_type}")
                
            except:
                # Eski modeller için preprocessing bilgisi olmayabilir
                pass
        
        # Eğer eski model ise manuel preprocessing ayarları göster
        if model_category == "custom" and not has_preprocessing_info:
            st.sidebar.warning("⚠️ Bu model eski bir model olduğu için preprocessing bilgileri mevcut değil.")
            
            # Örnek veriyi yükle
            sample_df = load_data()
            available_columns = [col for col in sample_df.columns if col != 'target']
            
            # Manuel olarak çıkarılacak sütunları seç
            st.sidebar.subheader("Manuel Preprocessing Ayarları")
            manual_columns_to_drop = st.sidebar.multiselect(
                "Bu modelin eğitiminde çıkarılan sütunları seçin",
                options=available_columns,
                default=[],
                help="Eğer bu model eğitilirken bazı sütunlar çıkarıldıysa, burada seçin."
            )
            
            if manual_columns_to_drop:
                columns_to_drop = manual_columns_to_drop
                st.sidebar.success(f"Seçilen {len(columns_to_drop)} sütun tahmin verisinden çıkarılacak.")
            
            # Model için preprocessing bilgilerini güncelleme seçeneği
            if st.sidebar.button("Bu Bilgileri Modele Kaydet", help="Bu preprocessing ayarlarını modele kalıcı olarak kaydeder."):
                try:
                    # Modeli yeniden yükle
                    model, model_info, metrics, old_preprocessing_info, old_scaler = load_model(selected_model_name, directory=f"models/{model_category}")
                    
                    # Yeni preprocessing bilgilerini hazırla
                    updated_preprocessing_info = {
                        "columns_to_drop": columns_to_drop,
                        "target_column": "target",  # Varsayılan
                        "feature_names": [col for col in available_columns if col not in columns_to_drop],
                        "test_size": 0.2,  # Varsayılan
                        "scaler_type": "StandardScaler"  # Varsayılan
                    }
                    
                    # Modeli güncellenen bilgilerle yeniden kaydet
                    save_model(model, model_info, metrics, selected_model_name, 
                             directory=f"models/{model_category}", 
                             preprocessing_info=updated_preprocessing_info,
                             scaler=old_scaler)
                    
                    st.sidebar.success("Model preprocessing bilgileri güncellendi! Sayfayı yenileyin.")
                    
                except Exception as e:
                    st.sidebar.error(f"Model güncellenirken hata: {str(e)}")
        
        elif model_category == "ready":
            # Hazır modeller için demo uyarısı
            if "demo" in selected_model_name:
                st.sidebar.info("Demo modelleri için preprocessing bilgisi mevcut değil.")
        
        # Tahmin için veri al
        st.header("Tahmin için Veri")
        
        # Model scaler'ını al (eğer custom model ise)
        model_scaler_for_input = None
        if model_category == "custom" and has_preprocessing_info:
            try:
                _, _, _, _, model_scaler_for_input = load_model(selected_model_name, directory=f"models/{model_category}")
            except:
                pass
        
        example_data, true_label, feature_names = get_prediction_data(model_feature_names, columns_to_drop, model_scaler_for_input)
        
        if example_data is not None and len(example_data) > 0:
            st.subheader("Tahmin Edilecek Veri")
            st.dataframe(example_data, use_container_width=True)
            
            # Tahmin butonu
            predict_button = st.button("Tahmin Et", type="primary")
            
            if predict_button:
                st.header("Tahmin Sonuçları")
                
                # Demo modeller için rastgele tahmin
                if is_demo:
                    # Rastgele bir olasılık üret
                    np.random.seed(int(datetime.now().timestamp()) % 10000)
                    probability = np.random.beta(5, 2)  # Genellikle yüksek olasılık üretir
                    prediction = 1 if probability > threshold else 0
                    
                    st.subheader("Model Tahmini")
                    
                    # Sınıf etiketi
                    label = "Malignant (Kötü Huylu)" if prediction == 1 else "Benign (İyi Huylu)"
                    label_color = "red" if prediction == 1 else "green"
                    
                    # Tahmin sonucunu göster
                    st.markdown(f"<h3 style='color: {label_color};'>Tahmin: {label}</h3>", unsafe_allow_html=True)
                    
                    # Olasılık çubuğu
                    st.subheader("Tahmin Olasılığı")
                    st.progress(float(probability))
                    st.write(f"Kötü Huylu (Malignant) Olasılığı: {probability:.2%}")
                    
                    # Eğer gerçek etiket varsa göster
                    if true_label is not None:
                        true_class = "Malignant (Kötü Huylu)" if true_label == 1 else "Benign (İyi Huylu)"
                        st.write(f"**Gerçek Sınıf:** {true_class}")
                    
                    # Demo modeller için rassal özellik önemi
                    st.subheader("Özellik Önemi (Demo)")
                    
                    # Rastgele önem puanları oluştur
                    np.random.seed(42)  # Tekrarlanabilirlik için
                    feature_importance = pd.DataFrame({
                        "feature": feature_names,
                        "importance": np.random.rand(len(feature_names))
                    }).sort_values("importance", ascending=False)
                    
                    # Özellik önemini görselleştir
                    fig = plt.figure(figsize=(10, 6))
                    plt.barh(feature_importance['feature'], feature_importance['importance'])
                    plt.xlabel('Önem Derecesi')
                    plt.ylabel('Özellik')
                    plt.title('Özellik Önemi')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                else:
                    try:
                        # Gerçek modeli yükle
                        model, model_info, metrics, preprocessing_info, model_scaler = load_model(selected_model_name, directory=f"models/{model_category}")
                        
                        # Veriyi modelin beklediği formata getir (preprocessing uygula)
                        example_data_processed = apply_model_preprocessing(example_data, preprocessing_info, model_scaler)
                        
                        if example_data_processed.shape[0] > 0:
                            st.info(f"Veri preprocessing uygulandı. Boyut: {example_data_processed.shape}")
                            
                            # Eğitimde çıkarılan sütunlar hakkında bilgi ver
                            if preprocessing_info and preprocessing_info.get("columns_to_drop"):
                                dropped_cols = preprocessing_info["columns_to_drop"]
                                st.info(f"Model eğitiminde çıkarılan sütunlar tahmin verisinden de çıkarıldı: {', '.join(dropped_cols)}")
                        else:
                            st.warning("Preprocessing uygulandıktan sonra veri boyutu sıfır oldu.")
                        
                        # Modelin türüne göre tahmin yap
                        if model_info["type"] == "neural_network":
                            probabilities = model.predict(example_data_processed).ravel()
                            prediction = (probabilities > threshold).astype(int)
                        else:  # classical model
                            prediction = model.predict(example_data_processed)
                            
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(example_data_processed)[:, 1]
                            elif hasattr(model, "decision_function"):
                                # SVM gibi modeller için karar fonksiyonunu olasılığa dönüştür
                                decision_values = model.decision_function(example_data_processed)
                                probabilities = 1 / (1 + np.exp(-decision_values))
                            else:
                                probabilities = prediction.astype(float)
                        
                        st.subheader("Model Tahmini")
                        
                        # Sınıf etiketi
                        label = "Malignant (Kötü Huylu)" if prediction[0] == 1 else "Benign (İyi Huylu)"
                        label_color = "red" if prediction[0] == 1 else "green"
                        
                        # Tahmin sonucunu göster
                        st.markdown(f"<h3 style='color: {label_color};'>Tahmin: {label}</h3>", unsafe_allow_html=True)
                        
                        # Olasılık çubuğu
                        st.subheader("Tahmin Olasılığı")
                        st.progress(float(probabilities[0]))
                        st.write(f"Kötü Huylu (Malignant) Olasılığı: {float(probabilities[0]):.2%}")
                        
                        # Eğer gerçek etiket varsa göster
                        if true_label is not None:
                            true_class = "Malignant (Kötü Huylu)" if true_label == 1 else "Benign (İyi Huylu)"
                            st.write(f"**Gerçek Sınıf:** {true_class}")
                        
                        # SHAP değerleri yerine özellik önemi göster (eğer model destekliyorsa)
                        if model_info["type"] != "neural_network" and model_info["type"] not in ["knn"]:
                            from utils.model_utils import calculate_feature_importance
                            
                            st.subheader("Özellik Önemi")
                            
                            # Özellik önemini hesapla
                            feature_importance = calculate_feature_importance(
                                model, 
                                feature_names, 
                                "classical" if model_info["type"] != "neural_network" else "neural_network"
                            )
                            
                            # Özellik önemini görselleştir
                            importance_fig = plot_feature_importance(feature_importance)
                            st.pyplot(importance_fig)
                        
                    except Exception as e:
                        st.error(f"Tahmin yapılırken bir hata oluştu: {str(e)}")
                        st.exception(e)
            
    else:
        st.error("Seçilen model bulunamadı.")

except Exception as e:
    st.error(f"Modeller listelenirken bir hata oluştu: {str(e)}")
    st.exception(e)
