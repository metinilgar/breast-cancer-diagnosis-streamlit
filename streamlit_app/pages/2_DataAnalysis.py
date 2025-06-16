import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Modülleri import et
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data, check_data_quality, detect_outliers
from utils.visualizations import (
    plot_histograms, plot_correlation_heatmap, plot_pairplot, 
    plot_feature_distribution, plot_feature_boxplots, 
    plot_interactive_scatter, plot_interactive_3d_scatter, plot_pca_visualization
)

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Veri Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana başlık
st.title("📊 Veri Analizi ve Keşif")

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

# Hedef sütun seçimi
if df is not None:
    # Eğer örnek veri seti kullanılıyorsa varsayılan hedef sütun "target" olacak
    default_target_index = 0  # Varsayılan olarak boş seçenek
    
    if data_source == "Örnek Veri Seti" and "target" in df.columns:
        # Örnek veri seti için "target" sütununu seçili hale getir
        default_target_index = df.columns.tolist().index("target") + 1  # +1 çünkü boş seçenek ekledik
    
    target_column = st.sidebar.selectbox(
        "Hedef Sütunu Seçin",
        options=[""] + df.columns.tolist(),  # Boş seçenek eklendi
        index=default_target_index  # Örnek veri setiyse target, değilse boş
    )
    
    # Hedef sütun seçilmediğinde uyarı göster
    if not target_column:
        st.sidebar.warning("⚠️ Lütfen hedef sütunu seçin. Aksi takdirde varsayılan hedef sütun kullanılacak!")
    
    # Analiz modülü seçimi
    analysis_module = st.sidebar.selectbox(
        "Analiz Modülü",
        ["Veri Önizleme", "Temel İstatistikler", "Dağılım Grafikleri", 
         "Korelasyon Analizi", "Özellik Analizi", "Aykırı Değer Analizi", "PCA Görselleştirme"]
    )
    
    # Analiz modülüne göre içerik göster
    if analysis_module == "Veri Önizleme":
        st.header("Veri Seti Önizleme")
        
        # Veri seti bilgisi
        st.markdown(f"**Satır Sayısı:** {df.shape[0]}, **Sütun Sayısı:** {df.shape[1]}")
        
        # Veri seti önizleme
        st.subheader("İlk 10 Satır")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Sütun bilgileri
        st.subheader("Sütun Bilgileri")
        
        # Sütun tiplerini göster
        col_types = pd.DataFrame({
            'Sütun': df.columns,
            'Tip': df.dtypes.astype(str),
            'Null Sayısı': df.isnull().sum(),
            'Null Yüzdesi': (df.isnull().sum() / len(df) * 100).round(2),
            'Benzersiz Değer Sayısı': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(col_types, use_container_width=True)
        
        # Eksik değerlerin %40'ından fazlası olan sütunları göster
        missing_percentage = df.isnull().mean()
        problematic_cols = missing_percentage[missing_percentage > 0.4].index.tolist()
        
        if problematic_cols:
            st.warning(f"⚠️ Aşağıdaki sütunlarda %40'tan fazla eksik veri var ve ön işleme sırasında silinecek (hedef sütun hariç):\n{', '.join(problematic_cols)}")
        
        # Hedef sütun seçilmediyse veya eksik verisi çok olan sütunsa uyarı göster
        if target_column in problematic_cols:
            st.error(f"⚠️ Seçilen hedef sütun '{target_column}' eksik verilerin çok olduğu bir sütun, ancak ön işleme sırasında korunacak.")
        
    elif analysis_module == "Temel İstatistikler":
        st.header("Temel İstatistiksel Analiz")
        
        # Veri kalitesi kontrolü
        quality_report = check_data_quality(df)
        
        # İstatistikler
        st.subheader("Sayısal Verilerin İstatistikleri")
        st.dataframe(df.describe().T, use_container_width=True)
        
        # Sınıf dağılımı
        if target_column:  # Hedef sütun seçildiyse
            if target_column in df.columns:
                st.subheader(f"Hedef Sınıf Dağılımı ({target_column})")
                target_counts = df[target_column].value_counts()
                
                # Pasta grafiği
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title(f'{target_column} Sınıf Dağılımı')
                st.pyplot(fig)
                
                # Sayısal olarak da göster
                st.dataframe(pd.DataFrame({
                    'Sınıf': target_counts.index,
                    'Sayı': target_counts.values,
                    'Yüzde': (target_counts.values / target_counts.sum() * 100).round(2)
                }), use_container_width=True)
            else:
                st.error(f"Seçilen hedef sütun '{target_column}' veri setinde bulunamadı.")
        else:
            st.warning("Lütfen hedef sütun seçin. Hedef sütun olmadan sınıf dağılımı gösterilemiyor.")
            
    elif analysis_module == "Dağılım Grafikleri":
        st.header("Özellik Dağılımları")
        
        # Sayısal özellikleri seç
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Seçili özellikler
        selected_features = st.multiselect(
            "Gösterilecek Özellikleri Seçin",
            options=numerical_cols,
            default=numerical_cols[:min(6, len(numerical_cols))]
        )
        
        if selected_features:
            # Histogram grafiği
            st.subheader("Histogram Grafikleri")
            fig = plot_histograms(df, selected_features)
            st.pyplot(fig)
            
            # Özellik dağılımları - hedef sınıfa göre
            if target_column and target_column in df.columns:
                st.subheader("Özellik Dağılımları (Sınıfa Göre)")
                
                selected_feature = st.selectbox(
                    "Özellik Seçin",
                    options=selected_features
                )
                
                fig = plot_feature_distribution(df, selected_feature, target_column)
                st.pyplot(fig)
                
                # Kutu grafikleri
                st.subheader("Kutu Grafikleri (Sınıfa Göre)")
                fig = plot_feature_boxplots(df, selected_features, target_column)
                st.pyplot(fig)
            elif not target_column:
                st.warning("Hedef sınıfa göre dağılımları görmek için lütfen hedef sütunu seçin.")
            else:
                st.error(f"Seçilen hedef sütun '{target_column}' veri setinde bulunamadı.")
                
        else:
            st.warning("Lütfen en az bir özellik seçin.")
            
    elif analysis_module == "Korelasyon Analizi":
        st.header("Korelasyon Analizi")
        
        # Korelasyon ısı haritası
        st.subheader("Özellikler Arası Korelasyon Isı Haritası")
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)
        
        # Çift grafik görselleştirme
        st.subheader("Çiftli Özellik Görselleştirmesi")
        
        # Sayısal özellikleri seç
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Rastgele 5 özellik seç (veya daha az, eğer 5'ten az özellik varsa)
        default_features = numerical_cols[:min(5, len(numerical_cols))]
        if target_column and target_column in default_features:
            default_features.remove(target_column)
        
        selected_features = st.multiselect(
            "Gösterilecek Özellikleri Seçin (max. 5 önerilir)",
            options=numerical_cols,
            default=default_features[:min(3, len(default_features))]
        )
        
        if len(selected_features) > 1:
            # Örnek sayısını sınırla (büyük veri setleri için)
            sample_size = st.slider("Örnek Sayısı", min_value=100, max_value=min(1000, len(df)), value=min(500, len(df)))
            
            # Hedef sütun seçilmediyse uyarı göster
            effective_target = target_column if target_column and target_column in df.columns else None
            if not effective_target:
                st.warning("Hedef sütun seçilmediği için sınıf renklendirilmesi olmayacaktır.")
                
            fig = plot_pairplot(df, effective_target, sample_size, 
                               selected_features + [effective_target] if effective_target and effective_target not in selected_features else selected_features)
            st.pyplot(fig)
        else:
            st.warning("Çiftli görselleştirme için en az 2 özellik seçin.")
            
    elif analysis_module == "Özellik Analizi":
        st.header("İnteraktif Özellik Analizi")
        
        # Sayısal özellikleri seç
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # 2D Dağılım grafiği
        st.subheader("2D Dağılım Grafiği")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X Ekseni İçin Özellik", options=numerical_cols, index=0)
        
        with col2:
            y_feature = st.selectbox("Y Ekseni İçin Özellik", options=numerical_cols, index=min(1, len(numerical_cols)-1))
        
        color_feature = st.selectbox(
            "Renk Kodlaması İçin Özellik", 
            options=["Yok"] + df.columns.tolist(), 
            index=df.columns.tolist().index(target_column)+1 if target_column and target_column in df.columns else 0
        )
        color_col = None if color_feature == "Yok" else color_feature
        
        # Dağılım grafiğini göster
        fig = plot_interactive_scatter(df, x_feature, y_feature, color_col)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Dağılım grafiği (3 özellik seçilirse)
        st.subheader("3D Dağılım Grafiği")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_feature_3d = st.selectbox("X Ekseni", options=numerical_cols, index=0)
        
        with col2:
            y_feature_3d = st.selectbox("Y Ekseni", options=numerical_cols, index=min(1, len(numerical_cols)-1))
            
        with col3:
            z_feature_3d = st.selectbox("Z Ekseni", options=numerical_cols, index=min(2, len(numerical_cols)-1))
        
        color_feature_3d = st.selectbox(
            "Renk Kodlaması", 
            options=["Yok"] + df.columns.tolist(),
            index=df.columns.tolist().index(target_column)+1 if target_column and target_column in df.columns else 0
        )
        color_col_3d = None if color_feature_3d == "Yok" else color_feature_3d
        
        # 3D Dağılım grafiğini göster
        fig = plot_interactive_3d_scatter(df, x_feature_3d, y_feature_3d, z_feature_3d, color_col_3d)
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_module == "Aykırı Değer Analizi":
        st.header("Aykırı Değer Analizi")
        
        # Sayısal özellikleri seç
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        # Seçili özellikler
        selected_features = st.multiselect(
            "Analiz Edilecek Özellikleri Seçin",
            options=numerical_cols,
            default=numerical_cols[:min(5, len(numerical_cols))]
        )
        
        # Aykırı değer tespiti yöntemi
        outlier_method = st.radio("Aykırı Değer Tespit Yöntemi", options=["IQR", "Z-Score"])
        
        if selected_features:
            # Aykırı değerleri tespit et
            outliers = detect_outliers(df, outlier_method.lower(), selected_features)
            
            # Özelliklere göre aykırı değer sayısı
            outlier_counts = {col: len(indices) for col, indices in outliers.items()}
            
            st.subheader("Aykırı Değer Sayıları")
            
            # Çubuk grafiği
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(outlier_counts.keys(), outlier_counts.values())
            
            # Çubukların üzerine sayıları ekle
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.title(f'Özellik Bazında Aykırı Değer Sayısı ({outlier_method} yöntemi)')
            plt.xlabel('Özellik')
            plt.ylabel('Aykırı Değer Sayısı')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Aykırı değerleri göster
            st.subheader("Aykırı Değerler")
            
            selected_feature_outlier = st.selectbox(
                "Aykırı Değerleri Gösterilecek Özelliği Seçin",
                options=selected_features
            )
            
            if outlier_counts[selected_feature_outlier] > 0:
                outlier_indices = outliers[selected_feature_outlier]
                st.dataframe(df.loc[outlier_indices, [selected_feature_outlier] + [col for col in df.columns if col != selected_feature_outlier][:5]], use_container_width=True)
                
                # Kutu grafiği
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=df[selected_feature_outlier], ax=ax)
                plt.title(f'{selected_feature_outlier} Kutu Grafiği')
                st.pyplot(fig)
            else:
                st.info(f"{selected_feature_outlier} özelliğinde aykırı değer bulunamadı.")
        else:
            st.warning("Lütfen en az bir özellik seçin.")
            
    elif analysis_module == "PCA Görselleştirme":
        st.header("Temel Bileşen Analizi (PCA) Görselleştirmesi")
        
        # Sayısal özellikleri seç
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        # Seçili özellikler
        selected_features = st.multiselect(
            "PCA için Özellikleri Seçin",
            options=numerical_cols,
            default=numerical_cols[:min(10, len(numerical_cols))]
        )
        
        if len(selected_features) >= 2:
            # PCA boyutunu seç
            pca_dimensions = st.radio("PCA Görselleştirme Boyutu", options=["2D", "3D"])
            n_components = 2 if pca_dimensions == "2D" else 3
            
            if target_column and target_column in df.columns:
                # PCA görselleştirme
                X = df[selected_features].values
                
                # Hedef değişkeni al
                y = df[target_column]
                
                # Hedef değişken kategorik mi kontrol et (object, category veya az sayıda benzersiz değer)
                if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y) or len(y.unique()) <= 5:
                    # Kategorik değerleri 0 ve 1'e dönüştür
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.info(f"Hedef sütun '{target_column}' kategorik olduğu için dönüştürüldü. Eşleşmeler: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                else:
                    # Sayısal değer zaten, array'e dönüştür
                    y = y.values
                
                fig = plot_pca_visualization(X, y, n_components)
                st.pyplot(fig)
                
                # PCA açıklaması
                st.markdown("""
                **PCA (Temel Bileşen Analizi) Nedir?**
                
                Temel Bileşen Analizi, yüksek boyutlu verileri daha düşük boyutlu bir uzaya dönüştüren bir boyut indirgeme tekniğidir.
                Bu görselleştirme, verilerin en çok varyans gösterdiği yönleri (temel bileşenleri) kullanarak verilerinizi 2D veya 3D olarak gösterir.
                
                - Birbirine yakın noktalar, orijinal yüksek boyutlu uzayda da benzer özelliklere sahiptir.
                - Farklı renkler farklı hedef sınıfları temsil eder.
                - Temel bileşenlerin yanındaki yüzde değerleri, o bileşenin toplam varyansın ne kadarını açıkladığını gösterir.
                """)
            elif not target_column:
                st.warning("Hedef sütun seçilmediğinden PCA görselleştirmesi tam olarak yapılamıyor. Lütfen hedef sütun seçin.")
            else:
                st.error(f"Seçilen hedef sütun '{target_column}' veri setinde bulunamadı.")
        else:
            st.warning("PCA görselleştirmesi için en az 2 özellik seçmelisiniz.")
            
else:
    st.error("Veri yüklenemedi. Lütfen veri kaynağınızı kontrol edin.")
