import streamlit as st

# Başlık ve sayfa konfigürasyonu
st.set_page_config(
    page_title="Ana Sayfa",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana başlık
st.title("🏠 Meme Kanseri Teşhis Platformu: Detaylı Rehber")

# Platform vizyonu
st.markdown("""
### 🌟 Platform Vizyonu ve Misyonu

Bu gelişmiş yapay zeka platformu, meme kanseri teşhis süreçlerinde sağlık profesyonellerine, araştırmacılara ve veri bilimcilere 
kapsamlı destek sağlamak amacıyla tasarlanmıştır. Wisconsin Breast Cancer veri seti ve benzeri tıbbi veri setleri üzerinde 
çalışan platform, 8'dan fazla farklı makine öğrenmesi algoritmasını ve modern derin öğrenme tekniklerini bir araya getirerek, 
kullanıcıların hem hazır modelleri incelemesini hem de kendi özel modellerini geliştirmesini sağlar.

**🎯 Ana Hedeflerimiz:**
- Tıbbi veri analizi süreçlerini hızlandırmak
- Makine öğrenmesi eğitimi için interaktif platform sağlamak
- Araştırma ve geliştirme süreçlerini hızlandırmak
- Yapay zeka teknolojilerinin tıp alanındaki potansiyelini göstermek
""")

# Platform modülleri
st.subheader("🔧 Platform Modülleri ve Yetenekleri")

# Gelişmiş özellikler
feature_tab1, feature_tab2, feature_tab3 = st.tabs([
    "📊 Veri Analizi Modülü", 
    "🤖 Model Geliştirme Modülü", 
    "🔬 Değerlendirme ve Tahmin Modülü"
])

with feature_tab1:
    st.markdown("""
    ### 📊 Kapsamlı Veri Analizi ve Görselleştirme
    
    #### **Temel Veri Analizi**
    - ✅ **Veri Kalitesi Kontrolü**: Eksik veri tespiti, aykırı değer analizi
    - ✅ **İstatistiksel Özetler**: Tanımlayıcı istatistikler, dağılım analizi
    - ✅ **Veri Tipi Analizi**: Kategorik/sayısal veri ayrımı, benzersiz değer sayıları
    
    #### **Gelişmiş Görselleştirmeler**
    - 📈 **Histogram ve Dağılım Grafikleri**: KDE ile birleştirilmiş dağılım analizi
    - 🔥 **Korelasyon Isı Haritası**: Özellikler arası ilişki analizi
    - 🎯 **Çiftli Özellik Grafikleri**: Pairplot ve scatter matrix analizi
    - 📦 **Kutu Grafikleri**: Sınıflara göre özellik dağılımları
    
    #### **Boyut Azaltma ve Keşif**
    - 🧮 **PCA Analizi**: 2D/3D temel bileşen görselleştirmesi
    - 🌟 **t-SNE Görselleştirmesi**: Non-linear boyut azaltma
    - 🔍 **İnteraktif 3D Grafikler**: Plotly ile etkileşimli keşif
    
    #### **Aykırı Değer Tespiti**
    - 📊 **IQR Yöntemi**: Çeyreklik açıklık yöntemi
    - 📈 **Z-Score Analizi**: Standart sapma tabanlı tespit
    - 🎯 **Görsel Tespit**: Kutu grafikleri ile aykırı değer görselleştirmesi
    """)

with feature_tab2:
    st.markdown("""
    ### 🤖 Gelişmiş Model Geliştirme Ortamı
    
    #### **Klasik Makine Öğrenmesi Algoritmaları**
    - 🌳 **Random Forest**: Ensemble learning, feature importance
    - 📈 **Logistic Regression**: Linear model, probabilistic output
    - 🎯 **Support Vector Machine (SVM)**: Kernel tricks, margin optimization
    - 👥 **K-Nearest Neighbors (KNN)**: Instance-based learning
    - 🌿 **Decision Tree**: Interpretable tree-based models
    - 🚀 **Gradient Boosting**: XGBoost style boosting algorithms
    - 🧠 **Multi-Layer Perceptron**: Classical neural network
    
    #### **Derin Öğrenme Modelleri**
    - 🔬 **Custom Neural Networks**: TensorFlow/Keras tabanlı
    - ⚙️ **Flexible Architecture**: 1-5 katman arası özelleştirme
    - 🎛️ **Hyperparameter Tuning**: Batch size, learning rate, dropout
    - 📊 **Training Monitoring**: Real-time loss ve accuracy tracking
    
    #### **Veri Ön İşleme Seçenekleri**
    - 📏 **StandardScaler**: Z-score normalizasyonu
    - 📐 **MinMaxScaler**: 0-1 arası ölçekleme
    - 🚫 **NoScaler**: Ham veri ile çalışma
    - 🗑️ **Feature Selection**: Manuel özellik seçimi ve çıkarma
    
    #### **Model Optimizasyonu**
    - 🎯 **Automated Evaluation**: Cross-validation ve test split
    - 💾 **Model Persistence**: Joblib ile model kaydetme
    - 📈 **Performance Tracking**: Comprehensive metrics collection
    """)

with feature_tab3:
    st.markdown("""
    ### 🔬 Model Değerlendirme ve Tahmin Sistemi
    
    #### **Performans Metrikleri**
    - ✅ **Accuracy**: Genel doğruluk oranı
    - 🎯 **Precision**: Pozitif tahmin kesinliği
    - 📊 **Recall (Sensitivity)**: Gerçek pozitif yakalama oranı
    - ⚖️ **F1-Score**: Precision ve recall harmonik ortalaması
    - 📈 **ROC-AUC**: Receiver Operating Characteristic curve
    - 🔢 **Confusion Matrix**: Detaylı sınıflandırma analizi
    
    #### **Model Karşılaştırma**
    - 📊 **Side-by-Side Comparison**: Çoklu model performans karşılaştırması
    - 🎨 **Visual Analytics**: ROC curves, bar charts, heatmaps
    - 📋 **Detailed Reports**: Comprehensive model evaluation reports
    - 🏆 **Ranking System**: Performance-based model ranking
    
    #### **Gerçek Zamanlı Tahmin**
    - ⚡ **Instant Predictions**: Single sample prediction
    - 📊 **Batch Processing**: Multiple sample processing
    - 🎯 **Confidence Scores**: Prediction probability display
    - 📈 **Real-time Visualization**: Dynamic result presentation
    
    #### **Özellik Önem Analizi**
    - 🌟 **Feature Importance**: Tree-based model feature rankings
    - 📊 **Interactive Charts**: Plotly-based importance visualization
    - 🔍 **Feature Selection Guidance**: Data-driven feature recommendations
    - 📋 **Interpretability Reports**: Model decision explanation
    """)

st.markdown("---")

# Proje hakkında detaylar
st.subheader("🔬 Platform Mimarisi ve Veri Bilimi Yaklaşımı")

# Veri bilimi süreci
st.markdown("""
### 📊 End-to-End Veri Bilimi Pipeline'ı

Platform, gerçek dünya veri bilimi projelerinde kullanılan standart süreçleri takip eder:

**1. 📥 Veri Toplama ve Yükleme**
- Wisconsin Breast Cancer Dataset (default)
- CSV/Excel dosya yükleme desteği
- Otomatik veri tipi tespiti ve dönüşümü

**2. 🔍 Keşifsel Veri Analizi (EDA)**
- Eksik veri analizi ve temizleme stratejileri
- Outlier detection ve handling
- Feature distribution analysis
- Correlation analysis ve multicollinearity detection

**3. ⚙️ Veri Ön İşleme**
- Feature scaling (StandardScaler, MinMaxScaler)
- Feature selection ve engineering
- Train-test split optimization
- Class imbalance handling

**4. 🤖 Model Geliştirme**
- Multiple algorithm comparison
- Hyperparameter optimization
- Cross-validation strategies
- Ensemble methods implementation

**5. 📈 Model Değerlendirme**
- Comprehensive metrics calculation
- ROC curve analysis
- Confusion matrix interpretation
- Feature importance analysis

**6. 🚀 Model Deployment**
- Model serialization (Joblib)
- Real-time prediction interface
- Batch processing capabilities
- Model versioning system
""")

# Veri kaynağı bilgileri
st.subheader("📚 Veri Kaynakları ve Kalitesi")
st.info("""
**🔬 Wisconsin Breast Cancer Dataset**
- **Kaynak**: UCI Machine Learning Repository
- **Örneklem Boyutu**: 569 hasta verisi
- **Özellik Sayısı**: 30 sayısal özellik
- **Hedef Değişken**: Binary classification (Malignant/Benign)
- **Veri Kalitesi**: Eksik veri yok, yüksek kaliteli cleaned dataset

**📊 Özellik Kategorileri**:
- **Radius**: Hücre çekirdeği çapı özellikleri
- **Texture**: Gri değer standart sapması
- **Perimeter**: Çekirdek çevresi
- **Area**: Çekirdek alanı
- **Smoothness**: Yarıçap uzunluklarındaki lokal değişim
- **Compactness**: (perimeter² / area - 1.0)
- **Concavity**: Kontürün konkav kısımlarının şiddeti
- **Symmetry**: Simetri özellikleri
- **Fractal Dimension**: Kıyı şeridi yaklaşımı - 1

Her özellik için mean, standard error ve "worst" değerleri hesaplanmıştır.
""", icon="ℹ️")


# Yardım ve destek
st.warning("""
⚠️ **Önemli Etik ve Yasal Uyarı**: 

Bu platform **sadece eğitim, araştırma ve metodoloji geliştirme** amaçlarına yöneliktir. 
Kesinlikle aşağıdaki amaçlar için kullanılmamalıdır:

- ❌ Klinik tanı ve teşhis kararları
- ❌ Tedavi planlaması ve hasta yönetimi  
- ❌ Tıbbi gözetim ve screening
- ❌ Profesyonel tıbbi konsültasyon yerine geçecek kullanım

**📋 Tıbbi Sorumluluk Reddi**: Tüm tıbbi kararlar lisanslı sağlık profesyonelleri tarafından, 
uygun klinik değerlendirme ve standart teşhis protokolleri çerçevesinde alınmalıdır.
""")


# İletişim bilgileri
st.sidebar.header("📞 Destek ve İletişim")
st.sidebar.info(
    """
   
    **💻 GitHub**: [breast-cancer-ml-platform](https://github.com/metinilgar) \n
    **📧 E-posta**: contact@ilgarmutlu.com

    """
)

st.sidebar.markdown("---")
st.sidebar.caption("Metin Ilgar Mutlu")

