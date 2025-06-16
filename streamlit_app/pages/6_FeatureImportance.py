import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Modülleri import et
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_data, preprocess_data
from utils.model_utils import list_available_models, load_model, calculate_feature_importance
from utils.visualizations import (
    plot_feature_importance, plot_correlation_heatmap, 
    plot_interactive_feature_importance, plot_pca_visualization
)

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Özellik Önemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana başlık
st.title("📊 Özellik Önemi ve Veri İlişkileri")

# Veriyi yükle
df = load_data()

# Hedef sütunu ayır
X = df.drop('target', axis=1) if 'target' in df.columns else df
y = df['target'] if 'target' in df.columns else None

# Özellik isimlerini al
feature_names = list(X.columns)

# Sidebar - Analiz Seçenekleri
st.sidebar.header("Analiz Seçenekleri")

analysis_type = st.sidebar.radio(
    "Analiz Türü",
    ["Model Bazlı Özellik Önemi", "Korelasyon Analizi", "Özellik Dağılımları", "Boyut İndirgeme"]
)

if analysis_type == "Model Bazlı Özellik Önemi":
    st.header("Model Bazlı Özellik Önemi Analizi")
    
    # Analiz yöntemi seçimi
    method = st.radio(
        "Özellik Önemi Yöntemi",
        ["Random Forest İçin Hesapla", "Permütasyon Önemi"]
    )
    
    # Random Forest ile özellik önemi hesaplama
    if method == "Random Forest İçin Hesapla":
        st.subheader("Random Forest Özellik Önemi")
        
        # Model parametreleri
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Ağaç Sayısı", min_value=10, max_value=200, value=100, step=10)
        with col2:
            max_depth = st.slider("Maksimum Derinlik", min_value=2, max_value=20, value=10, step=1)
        
        # Random Forest modeli eğit
        with st.spinner("Random Forest modeli eğitiliyor..."):
            # Veriyi ön işle
            X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, 'target', test_size=0.3)
            
            # Modeli oluştur ve eğit
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            rf.fit(X_train, y_train)
            
            # Modelin performansını değerlendir
            train_score = rf.score(X_train, y_train)
            test_score = rf.score(X_test, y_test)
            
            # Skorları göster
            st.write(f"**Eğitim Seti Doğruluğu:** {train_score:.4f}")
            st.write(f"**Test Seti Doğruluğu:** {test_score:.4f}")
            
            # Özellik önemini hesapla
            importances = rf.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Özellik önemini görselleştir
            st.subheader("Özellik Önemi")
            
            # Statik ve interaktif grafikleri göster
            fig = plot_feature_importance(feature_importance)
            st.pyplot(fig)
            
            st.subheader("İnteraktif Özellik Önemi")
            interactive_fig = plot_interactive_feature_importance(feature_importance)
            st.plotly_chart(interactive_fig, use_container_width=True)
            
            # Özellik önemini tablo olarak göster
            st.subheader("Özellik Önemi Tablosu")
            st.dataframe(feature_importance, use_container_width=True)
    
    # Permütasyon önemi
    else:  # Permütasyon Önemi
        st.subheader("Permütasyon Önemi")
        st.markdown("""
        Permütasyon önemi, bir özelliğin model performansına katkısını ölçmek için kullanılan model-agnostik bir yöntemdir.
        Bu yöntem, bir özelliğin değerlerini rastgele karıştırarak o özelliğin model performansına etkisini ölçer.
        """)
        
        # Model seçimi
        model_type = st.selectbox(
            "Baz Model Seçimi",
            options=["Random Forest", "Logistic Regression"],
            index=0
        )
        
        # Permütasyon yineleme sayısı
        n_repeats = st.slider("Permütasyon Yineleme Sayısı", min_value=1, max_value=30, value=10, step=1)
        
        # Permütasyon önemi hesapla
        with st.spinner("Permütasyon önemi hesaplanıyor..."):
            # Veriyi ön işle
            X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, 'target', test_size=0.3)
            
            # Modeli oluştur ve eğit
            if model_type == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Logistic Regression
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42)
            
            model.fit(X_train, y_train)
            
            # Modelin performansını değerlendir
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Skorları göster
            st.write(f"**Eğitim Seti Doğruluğu:** {train_score:.4f}")
            st.write(f"**Test Seti Doğruluğu:** {test_score:.4f}")
            
            # Permütasyon önemi hesapla
            perm_importance = permutation_importance(
                model, X_test, y_test, 
                n_repeats=n_repeats, 
                random_state=42
            )
            
            # Özellik önemini DataFrame'e dönüştür
            perm_feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean
            }).sort_values('importance', ascending=False)
            
            # Özellik önemini görselleştir
            st.subheader("Permütasyon Önemi")
            
            # Statik ve interaktif grafikleri göster
            fig = plot_feature_importance(perm_feature_importance)
            st.pyplot(fig)
            
            st.subheader("İnteraktif Permütasyon Önemi")
            interactive_fig = plot_interactive_feature_importance(perm_feature_importance)
            st.plotly_chart(interactive_fig, use_container_width=True)
            
            # Özellik önemini tablo olarak göster
            st.subheader("Permütasyon Önemi Tablosu")
            st.dataframe(perm_feature_importance, use_container_width=True)
            
            # Permütasyon öneminin standart sapmasını görselleştir
            st.subheader("Permütasyon Önemi Standart Sapması")
            
            # Standart sapmaları DataFrame'e ekle
            perm_feature_importance['std'] = perm_importance.importances_std
            
            # Hata çubuklu grafik
            plt.figure(figsize=(12, 8))
            plt.errorbar(
                x=perm_feature_importance['importance'],
                y=perm_feature_importance['feature'],
                xerr=perm_feature_importance['std'],
                fmt='o',
                capsize=5
            )
            plt.xlabel('Permütasyon Önemi')
            plt.ylabel('Özellik')
            plt.title('Permütasyon Önemi ve Standart Sapma')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(plt)

elif analysis_type == "Korelasyon Analizi":
    st.header("Özellikler Arası Korelasyon Analizi")
    
    # Korelasyon ısı haritası
    st.subheader("Korelasyon Isı Haritası")
    corr_fig = plot_correlation_heatmap(X)
    st.pyplot(corr_fig)
    
    # En yüksek korelasyonlu özellikleri göster
    st.subheader("En Yüksek Korelasyonlu Özellik Çiftleri")
    
    # Korelasyon matrisini hesapla
    corr_matrix = X.corr().abs()
    
    # Üst üçgeni sıfırla (tekrarları önlemek için)
    corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)] = 0
    
    # En yüksek korelasyonlu çiftleri bul
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs > 0].reset_index()
    corr_pairs.columns = ['Özellik 1', 'Özellik 2', 'Korelasyon']
    
    # Korelasyon tablosunu göster
    st.dataframe(corr_pairs.head(15), use_container_width=True)
    
    # Seçili özellik çiftinin dağılım grafiği
    st.subheader("Özellik Çifti Dağılım Grafiği")
    
    # Özellik seçimi
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Birinci Özellik", options=feature_names, index=0)
    with col2:
        feature2 = st.selectbox("İkinci Özellik", options=feature_names, index=min(1, len(feature_names)-1))
    
    # Dağılım grafiği
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[feature1], X[feature2], c=y, cmap='viridis', alpha=0.7)
    
    # Regresyon çizgisi
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[feature1], X[feature2])
    plt.plot(X[feature1], intercept + slope*X[feature1], 'r', label=f'r = {r_value:.4f}')
    
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'{feature1} ve {feature2} Arasındaki İlişki')
    plt.legend()
    plt.colorbar(scatter, label='Hedef Sınıf')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)

elif analysis_type == "Özellik Dağılımları":
    st.header("Özellik Dağılımları ve Hedef İlişkisi")
    
    # Özellik seçimi
    selected_feature = st.selectbox("Özellik Seçin", options=feature_names)
    
    # Seçilen özelliğin dağılımı
    st.subheader(f"{selected_feature} Dağılımı (Sınıfa Göre)")
    
    # Kutu grafiği ve dağılım grafiği için 2 sütunlu düzen
    col1, col2 = st.columns(2)
    
    with col1:
        # Kutu grafiği
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=y, y=X[selected_feature])
        plt.xlabel('Hedef Sınıf (0: Benign, 1: Malignant)')
        plt.ylabel(selected_feature)
        plt.title(f'{selected_feature} Kutu Grafiği')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
    
    with col2:
        # Dağılım grafiği
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=selected_feature, hue='target', kde=True, bins=30)
        plt.xlabel(selected_feature)
        plt.ylabel('Yoğunluk')
        plt.title(f'{selected_feature} Dağılımı')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
    
    # Özelliğin istatistikleri
    st.subheader(f"{selected_feature} İstatistikleri")
    
    # Sınıf bazında istatistikler
    stats_df = df.groupby('target')[selected_feature].describe().T
    stats_df.columns = ['Benign (0)', 'Malignant (1)']
    st.dataframe(stats_df, use_container_width=True)
    
    # Özelliğin hedef sınıf ile ilişkisi
    st.subheader(f"{selected_feature} ve Hedef Sınıf İlişkisi")
    
    # Point-biserial korelasyon (sürekli ve ikili değişken arasında)
    from scipy import stats
    correlation, p_value = stats.pointbiserialr(X[selected_feature], y)
    
    st.write(f"**Point-Biserial Korelasyon:** {correlation:.4f}")
    st.write(f"**p-değeri:** {p_value:.4g}")
    
    if p_value < 0.05:
        if correlation > 0:
            st.write(f"**Sonuç:** {selected_feature} değeri arttıkça, malignant olma olasılığı artıyor.")
        else:
            st.write(f"**Sonuç:** {selected_feature} değeri arttıkça, malignant olma olasılığı azalıyor.")
    else:
        st.write(f"**Sonuç:** {selected_feature} ve hedef sınıf arasında istatistiksel olarak anlamlı bir ilişki bulunamadı.")
    
    # Tek değişkenli sınıflandırma performansı
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    # Tekli özellik için çapraz doğrulama
    cv_scores = cross_val_score(tree, X[[selected_feature]], y, cv=5)
    
    st.write(f"**Tek Değişkenli Sınıflandırma Doğruluğu:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

else:  # Boyut İndirgeme
    st.header("Boyut İndirgeme Analizi")
    
    # Boyut indirgeme yöntemi seçimi
    dim_reduction_method = st.radio(
        "Boyut İndirgeme Yöntemi",
        ["PCA (Temel Bileşen Analizi)", "t-SNE"]
    )
    
    if dim_reduction_method == "PCA (Temel Bileşen Analizi)":
        st.subheader("PCA Analizi")
        
        # PCA boyutu seçimi
        pca_dim = st.radio("PCA Görselleştirme Boyutu", options=["2D", "3D"])
        n_components = 2 if pca_dim == "2D" else 3
        
        # PCA görselleştirme
        pca_fig = plot_pca_visualization(X.values, y.values, n_components)
        st.pyplot(pca_fig)
        
        st.markdown("""
        **PCA (Temel Bileşen Analizi) Nedir?**
        
        Temel Bileşen Analizi, yüksek boyutlu verileri daha düşük boyutlu bir uzaya dönüştüren bir boyut indirgeme tekniğidir.
        Bu görselleştirme, verilerin en çok varyans gösterdiği yönleri (temel bileşenleri) kullanarak verilerinizi 2D veya 3D olarak gösterir.
        
        - Birbirine yakın noktalar, orijinal yüksek boyutlu uzayda da benzer özelliklere sahiptir.
        - Farklı renkler farklı hedef sınıfları temsil eder.
        - Temel bileşenlerin yanındaki yüzde değerleri, o bileşenin toplam varyansın ne kadarını açıkladığını gösterir.
        """)
        
        # Top 10 özelliğin PCA bileşenlerine katkısını göster
        st.subheader("PCA Bileşen Yükleri (Component Loadings)")
        
        from sklearn.decomposition import PCA
        
        # PCA hesapla
        pca = PCA(n_components=min(3, X.shape[1]))
        pca.fit(X)
        
        # Bileşen yüklerini DataFrame'e dönüştür
        component_names = [f"PC{i+1}" for i in range(pca.n_components_)]
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=component_names,
            index=feature_names
        )
        
        # Her bileşen için en önemli 10 özelliği göster
        for i, component in enumerate(component_names):
            st.write(f"**{component} İçin En Önemli 10 Özellik:**")
            
            # Mutlak değer olarak en büyük yükleri al
            top_features = loadings[component].abs().sort_values(ascending=False).head(10)
            
            # Grafik çiz
            plt.figure(figsize=(10, 6))
            colors = ['red' if loadings.loc[feat, component] < 0 else 'blue' for feat in top_features.index]
            plt.barh(top_features.index, top_features.values, color=colors)
            plt.xlabel(f'{component} Yükleri')
            plt.ylabel('Özellikler')
            plt.title(f'En Yüksek {component} Yükleri')
            plt.tight_layout()
            st.pyplot(plt)
        
        # Bileşenlerin açıkladığı varyans
        st.subheader("PCA Bileşenlerinin Açıkladığı Varyans")
        
        # Varyans grafiği
        plt.figure(figsize=(10, 6))
        plt.bar(component_names, pca.explained_variance_ratio_ * 100)
        plt.xlabel('Temel Bileşenler')
        plt.ylabel('Açıklanan Varyans Yüzdesi')
        plt.title('Her Bileşenin Açıkladığı Varyans Yüzdesi')
        plt.tight_layout()
        st.pyplot(plt)
        
        # Kümülatif varyans grafiği
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(component_names) + 1), np.cumsum(pca.explained_variance_ratio_ * 100), marker='o')
        plt.xlabel('Bileşen Sayısı')
        plt.ylabel('Kümülatif Açıklanan Varyans Yüzdesi')
        plt.title('Kümülatif Açıklanan Varyans')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
        
    else:  # t-SNE
        st.subheader("t-SNE Analizi")
        
        # t-SNE parametreleri
        col1, col2 = st.columns(2)
        with col1:
            perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30, step=5)
        with col2:
            n_iter = st.slider("İterasyon Sayısı", min_value=250, max_value=1000, value=300, step=50)
        
        # t-SNE görselleştirme
        with st.spinner("t-SNE hesaplanıyor... Bu işlem biraz zaman alabilir."):
            from utils.visualizations import plot_tsne_visualization
            
            tsne_fig = plot_tsne_visualization(X.values, y.values, perplexity=perplexity, n_iter=n_iter)
            st.pyplot(tsne_fig)
        
        st.markdown("""
        **t-SNE (t-Distributed Stochastic Neighbor Embedding) Nedir?**
        
        t-SNE, yüksek boyutlu verileri düşük boyutlu bir uzaya dönüştürmek için kullanılan bir boyut indirgeme tekniğidir.
        PCA'dan farklı olarak, t-SNE özellikle verilerdeki yerel benzerlikleri korumayı amaçlar.
        
        - Birbirine yakın noktalar, orijinal yüksek boyutlu uzayda da benzerdir.
        - Farklı renkler farklı hedef sınıfları temsil eder.
        - t-SNE, veri kümelerini ayırmada PCA'dan daha başarılı olabilir, ancak her çalıştırmada farklı sonuçlar üretebilir.
        
        **Parametreler:**
        - **Perplexity:** Yerel komşuluk boyutunu kontrol eder. Düşük değerler küçük yerel yapılara odaklanırken, 
          yüksek değerler daha geniş yapılara odaklanır.
        - **İterasyon Sayısı:** Algoritmanın çalışacağı adım sayısı. Daha yüksek değerler daha iyi sonuçlar verebilir,
          ancak hesaplama süresi artar.
        """)
        
        st.warning("""
        Not: t-SNE sonuçları her çalıştırmada farklı olabilir ve perplexity parametresine oldukça duyarlıdır.
        Ayrıca, t-SNE global yapıyı korumak yerine yerel yapıyı korumaya odaklandığından, 
        uzak noktalar arasındaki ilişkiler her zaman doğru şekilde temsil edilmeyebilir.
        """)
