import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import roc_curve, auc

# Modülleri import et
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data
from utils.model_utils import (
    list_available_models, load_model, evaluate_model,
    plot_confusion_matrix, plot_roc_curve, calculate_feature_importance,
    apply_model_preprocessing
)
from utils.visualizations import (
    plot_feature_importance, plot_metrics_comparison, plot_roc_comparison
)

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Hazır Modeller",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana başlık
st.title("🔍 Hazır Modelleri Keşfedin")

# Sayfa açıklaması
st.info("""
Bu sayfada hazır modeller ve kendi eğittiğiniz modelleri karşılaştırabilirsiniz. 
Modellerin performans metriklerini, ROC eğrilerini ve karmaşıklık matrislerini yan yana inceleyebilirsiniz.
Kapsamlı karşılaştırma için soldaki menüden birden fazla model seçmeniz önerilir.
""")

# Mevcut modelleri listele
try:
    available_models = list_available_models()
    
    # Eğer hiç model yoksa uyarı göster
    if len(available_models["ready"]) == 0 and len(available_models["custom"]) == 0:
        st.warning("""
        Hiç model bulunamadı! 
        'Model Eğitimi' sayfasını kullanarak kendi modellerinizi eğitebilirsiniz.
        """)
        st.stop()  # Sayfanın geri kalanını çalıştırmayı durdur
    
    # Sidebar - Model Seçimi
    st.sidebar.header("Model Seçimi")
    
    model_category = st.sidebar.radio(
        "Model Kategorisi",
        ["Tüm Modeller", "Hazır Modeller", "Kullanıcı Modelleri"]
    )
    
    # Kategoriye göre modelleri filtrele
    filtered_models = []
    if model_category == "Hazır Modeller":
        filtered_models = available_models["ready"]
    elif model_category == "Kullanıcı Modelleri":
        filtered_models = available_models["custom"]
    else:
        filtered_models = available_models["ready"] + available_models["custom"]
    
    # Model seçeneklerini hazırla
    model_options = [model["name"] for model in filtered_models]
    
    # Eğer hiç model yoksa hata mesajı göster
    if len(filtered_models) == 0:
        st.sidebar.warning(f"Seçilen kategoride ({model_category}) hiç model bulunamadı.")
    
    # Model seçimi - en azından 2 model seçmeye teşvik et
    selected_models = st.sidebar.multiselect(
        "Karşılaştırılacak Modelleri Seçin",
        options=model_options,
        default=model_options[:min(2, len(model_options))] if model_options else []
    )
    
    # Seçilen model sayısını göster
    if selected_models:
        st.sidebar.info(f"{len(selected_models)} model seçildi. Karşılaştırma için en az 2 model seçilmesi önerilir.")
    
    # Görüntüleme seçenekleri
    display_option = st.sidebar.radio(
        "Görüntüleme Seçeneği",
        ["Model Karşılaştırma", "Model Detayları"],
        index=0  # Varsayılan olarak Model Karşılaştırma seçili
    )
    
    # Ana sayfa içeriği
    if selected_models:
        # Seçili tüm modelleri değerlendir
        selected_model_details = []
        for model_name in selected_models:
            # Modelin bilgilerini bul
            model_info = next((model for model in filtered_models if model["name"] == model_name), None)
            
            if model_info:
                # Modelin hangi kategoride olduğunu bul
                category = "ready" if model_info in available_models["ready"] else "custom"
                
                try:
                    # Modeli yükle
                    try:
                        model, info, metrics, _, _ = load_model(model_name, directory=f"models/{category}")
                        
                        # Modelin detaylarını kaydet
                        selected_model_details.append({
                            "name": model_name,
                            "category": "Hazır Model" if category == "ready" else "Kullanıcı Modeli",
                            "type": info["type"],
                            "params": info.get("params", {}),
                            "metrics": metrics,
                            "model": model,
                            "model_info": info
                        })
                    except FileNotFoundError:
                        st.error(f"Model dosyası bulunamadı: {model_name}")
                except Exception as e:
                    st.error(f"Model '{model_name}' yüklenirken hata oluştu: {str(e)}")
        
        # Modelleri göster
        if selected_model_details:
            # Seçenekler
            if display_option == "Model Karşılaştırma":
                st.header("Model Karşılaştırma")
                
                # Karşılaştırma bölümleri için sekmeler
                comparison_tabs = st.tabs([
                    "Performans Metrikleri", 
                    "ROC Eğrileri", 
                    "Karmaşıklık Matrisleri",
                    "Model Parametreleri"
                ])
                
                # Tab 1: Performans Metrikleri
                with comparison_tabs[0]:
                    # Model metriklerini karşılaştır
                    st.subheader("Performans Metrik Karşılaştırması")
                    
                    models_metrics = {
                        model["name"]: model["metrics"] for model in selected_model_details
                    }
                    
                    # Çubuk grafiği
                    fig = plot_metrics_comparison(models_metrics)
                    st.pyplot(fig)
                    
                    # Metrik tablosu
                    st.subheader("Detaylı Metrik Tablosu")
                    
                    metrics_df_data = []
                    metrics_to_show = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                    metrics_labels = {
                        "accuracy": "Doğruluk", 
                        "precision": "Kesinlik", 
                        "recall": "Hassasiyet", 
                        "f1_score": "F1 Skoru", 
                        "roc_auc": "ROC AUC"
                    }
                    
                    for model_name, metrics in models_metrics.items():
                        model_metrics = {"Model": model_name}
                        for metric in metrics_to_show:
                            if metric in metrics and metrics[metric] is not None:
                                model_metrics[metrics_labels[metric]] = f"{metrics[metric]:.4f}"
                            else:
                                model_metrics[metrics_labels[metric]] = "N/A"
                        metrics_df_data.append(model_metrics)
                    
                    if metrics_df_data:
                        metrics_df = pd.DataFrame(metrics_df_data)
                        st.dataframe(metrics_df, use_container_width=True)
                
                # Tab 2: ROC Eğrileri
                with comparison_tabs[1]:
                    st.subheader("ROC Eğrisi Karşılaştırması")
                    
                    # Veri yükle
                    df = load_data()
                    X = df.drop('target', axis=1) if 'target' in df.columns else df
                    y = df['target'] if 'target' in df.columns else None
                    
                    if y is not None:
                        models_roc = {}
                        
                        for model_detail in selected_model_details:
                            try:
                                # Model için preprocessing bilgilerini ve scaler'ı al
                                model_category = "custom" if model_detail["category"] == "Kullanıcı Modeli" else "ready"
                                
                                # Preprocessing bilgilerini al
                                preprocessing_info = {}
                                scaler = None
                                
                                if model_category == "custom":
                                    try:
                                        _, _, _, preprocessing_info, scaler = load_model(model_detail["name"], directory=f"models/{model_category}")
                                    except:
                                        # Eski modeller için preprocessing bilgisi olmayabilir
                                        pass
                                
                                # Veriyi modelin beklediği formata getir
                                X_processed = apply_model_preprocessing(X, preprocessing_info, scaler)
                                
                                # Gerçek modeller için tahmin yap
                                if model_detail["type"] == "neural_network":
                                    y_pred_proba = model_detail["model"].predict(X_processed).ravel()
                                else:  # classical model
                                    if hasattr(model_detail["model"], "predict_proba"):
                                        y_pred_proba = model_detail["model"].predict_proba(X_processed)[:, 1]
                                    elif hasattr(model_detail["model"], "decision_function"):
                                        y_pred_proba = model_detail["model"].decision_function(X_processed)
                                    else:
                                        y_pred_proba = model_detail["model"].predict(X_processed)
                                
                                # ROC hesapla
                                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                # ROC verilerini kaydet
                                models_roc[model_detail["name"]] = (fpr, tpr, roc_auc)
                                
                            except Exception as e:
                                st.warning(f"{model_detail['name']} modeli için ROC hesaplanamadı: {str(e)}")
                        
                        # ROC eğrilerini çiz
                        if models_roc:
                            fig = plot_roc_comparison(models_roc)
                            st.pyplot(fig)
                        else:
                            st.error("Hiçbir model için ROC eğrisi hesaplanamadı.")
                    else:
                        st.error("ROC karşılaştırması için hedef değişken bulunamadı.")
                
                # Tab 3: Karmaşıklık Matrisleri
                with comparison_tabs[2]:
                    st.subheader("Karmaşıklık Matrisi Karşılaştırması")
                    
                    # Karmaşıklık matrislerini göster
                    model_cols = st.columns(min(3, len(selected_model_details)))
                    
                    for i, model_detail in enumerate(selected_model_details):
                        col_idx = i % 3
                        with model_cols[col_idx]:
                            st.subheader(f"{model_detail['name']}")
                            
                            if "confusion_matrix" in model_detail["metrics"]:
                                cm = np.array(model_detail["metrics"]["confusion_matrix"])
                                cm_fig = plot_confusion_matrix(cm)
                                st.pyplot(cm_fig)
                            else:
                                st.warning("Bu model için karmaşıklık matrisi mevcut değil.")
                
                # Tab 4: Model Parametreleri
                with comparison_tabs[3]:
                    st.subheader("Model Parametreleri Karşılaştırması")
                    
                    # Tablo oluştur
                    param_comparison = []
                    
                    for model_detail in selected_model_details:
                        model_params = {
                            "Model Adı": model_detail["name"],
                            "Model Türü": model_detail["type"],
                            "Kategori": model_detail["category"],
                            "Doğruluk": f"{model_detail['metrics']['accuracy']:.4f}"
                        }
                        
                        # Model parametrelerini ekle
                        for param_name, param_value in model_detail["params"].items():
                            # Eğer değer tuple veya list ise stringe çevir
                            if isinstance(param_value, (tuple, list)):
                                param_value = str(param_value)
                            model_params[param_name] = param_value
                        
                        param_comparison.append(model_params)
                    
                    # DataFrame oluştur ve göster
                    if param_comparison:
                        param_df = pd.DataFrame(param_comparison)
                        st.dataframe(param_df, use_container_width=True)
                    else:
                        st.error("Model parametreleri karşılaştırması için veri bulunamadı.")
            else:  # Model Detayları
                # Her bir modelin detaylarını göster
                for model_detail in selected_model_details:
                    st.header(f"{model_detail['name']} ({model_detail['category']})")
                    
                    # Model türü ve parametreleri
                    st.subheader("Model Bilgileri")
                    st.write(f"**Model Türü:** {model_detail['type']}")
                    
                    # Parametreleri göster
                    st.write("**Model Parametreleri:**")
                    st.json(model_detail["params"])
                    
                    # Model performansı
                    st.subheader("Model Performansı")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = model_detail["metrics"]
                    col1.metric("Doğruluk (Accuracy)", f"{metrics['accuracy']:.4f}")
                    col2.metric("Kesinlik (Precision)", f"{metrics['precision']:.4f}")
                    col3.metric("Hassasiyet (Recall)", f"{metrics['recall']:.4f}")
                    col4.metric("F1 Skoru", f"{metrics['f1_score']:.4f}")
                    
                    # ROC-AUC değeri
                    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
                        st.metric("ROC AUC Skoru", f"{metrics['roc_auc']:.4f}")
                    
                    # Karmaşıklık matrisi
                    if "confusion_matrix" in metrics:
                        st.subheader("Karmaşıklık Matrisi")
                        cm_fig = plot_confusion_matrix(np.array(metrics["confusion_matrix"]))
                        st.pyplot(cm_fig)
                    
                    # Veri yükle (gerçek tahminler için)
                    st.subheader("Gerçek Veri Üzerinde Tahmin")
                    
                    # Veri yükle
                    df = load_data()
                    
                    # Veriyi hazırla
                    X = df.drop('target', axis=1) if 'target' in df.columns else df
                    y = df['target'] if 'target' in df.columns else None
                    
                    if y is not None:
                        # Model için preprocessing bilgilerini ve scaler'ı al
                        model_category = "custom" if model_detail["category"] == "Kullanıcı Modeli" else "ready"
                        
                        # Preprocessing bilgilerini al
                        preprocessing_info = {}
                        scaler = None
                        
                        if model_category == "custom":
                            try:
                                _, _, _, preprocessing_info, scaler = load_model(model_detail["name"], directory=f"models/{model_category}")
                            except:
                                # Eski modeller için preprocessing bilgisi olmayabilir
                                pass
                        
                        # Veriyi modelin beklediği formata getir
                        X_processed = apply_model_preprocessing(X, preprocessing_info, scaler)
                        
                        # Model türüne göre tahmin yap
                        if model_detail["type"] == "neural_network":
                            y_pred_proba = model_detail["model"].predict(X_processed).ravel()
                            y_pred = (y_pred_proba > 0.5).astype(int)
                        else:  # classical model
                            y_pred = model_detail["model"].predict(X_processed)
                            
                            if hasattr(model_detail["model"], "predict_proba"):
                                y_pred_proba = model_detail["model"].predict_proba(X_processed)[:, 1]
                            elif hasattr(model_detail["model"], "decision_function"):
                                y_pred_proba = model_detail["model"].decision_function(X_processed)
                            else:
                                y_pred_proba = y_pred
                        
                        # ROC eğrisi
                        try:
                            fpr, tpr, _ = roc_curve(y, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            st.subheader("ROC Eğrisi")
                            fig = plot_roc_curve(y, y_pred_proba)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"ROC eğrisi çizilirken hata oluştu: {str(e)}")
                    
                    # Özellik önemi (eğer model destekliyorsa)
                    if (model_detail["type"] != "neural_network" and 
                        model_detail["type"] not in ["knn"]):
                        try:
                            st.subheader("Özellik Önemi")
                            
                            # Özellik isimlerini al
                            feature_cols = list(X.columns)
                            
                            # Özellik önemini hesapla
                            feature_importance = calculate_feature_importance(
                                model_detail["model"], 
                                feature_cols, 
                                "classical" if model_detail["type"] != "neural_network" else "neural_network"
                            )
                            
                            # Özellik önemini görselleştir
                            importance_fig = plot_feature_importance(feature_importance)
                            st.pyplot(importance_fig)
                        except Exception as e:
                            st.error(f"Özellik önemi gösterilirken hata oluştu: {str(e)}")
                    
                    st.markdown("---")
        
        else:
            st.warning("Seçili modeller yüklenemedi veya mevcut değil.")
    
    else:
        st.info("Lütfen sidebar'dan incelemek istediğiniz modelleri seçin.")

except Exception as e:
    st.error(f"Modeller listelenirken bir hata oluştu: {str(e)}")
    st.exception(e)
