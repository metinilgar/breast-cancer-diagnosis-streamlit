import streamlit as st
import pandas as pd
import numpy as np
import os

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Meme Kanseri Teşhis Uygulaması",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana sayfa başlığı
st.title("🩺 Yapay Zeka Destekli Meme Kanseri Teşhis Platformu")

# Kullanıcı modelleri ve hazır modeller için klasör oluşturma
os.makedirs("models/ready", exist_ok=True)
os.makedirs("models/custom", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Platform tanıtımı
st.markdown("""
### 🚀 Gelişmiş Tıbbi Görüntü Analizi Platformu

Bu uygulama, meme kanseri teşhisinde sağlık profesyonellerine destek sağlamak amacıyla geliştirilmiş kapsamlı bir yapay zeka platformudur. 
Modern makine öğrenmesi ve derin öğrenme teknolojilerini kullanarak, hücre görüntülerinden elde edilen özellikler üzerinden 
kanser hücrelerinin iyi huylu (benign) veya kötü huylu (malignant) olup olmadığını tahmin etmeye yardımcı olur.

**🎯 Platform Misyonu:** Tıbbi teşhis süreçlerinde yapay zeka teknolojilerinin etkin kullanımını desteklemek ve sağlık profesyonellerine 
karar verme süreçlerinde objektif, hızlı ve güvenilir araçlar sunmaktır.
""")

# Platform özellikleri
st.subheader("🔧 Platform Özellikleri")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📊 **Kapsamlı Veri Analizi**
    - İstatistiksel analiz ve görselleştirme
    - Korelasyon analizi ve özellik dağılımları
    - Etkileşimli 2D/3D görselleştirmeler
    - PCA ve t-SNE boyut azaltma
    - Aykırı değer tespiti ve kalite kontrolü
    """)

with col2:
    st.markdown("""
    #### 🤖 **Gelişmiş Model Eğitimi**
    - 7+ farklı makine öğrenmesi algoritması
    - Yapay sinir ağları (TensorFlow/Keras)
    - Hiperparametre optimizasyonu
    - Otomatik model değerlendirmesi
    - Özel veri seti desteği
    """)

with col3:
    st.markdown("""
    #### 🔍 **Akıllı Model Analizi**
    - Model performans karşılaştırması
    - ROC eğrisi ve metrik analizi
    - Özellik önem değerlendirmesi
    - Gerçek zamanlı tahmin sistemi
    - Karmaşıklık matrisi görselleştirmesi
    """)

# Kullanım senaryoları
st.subheader("🎯 Kullanım Senaryoları")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    st.markdown("""
    #### 🏥 **Sağlık Profesyonelleri İçin**
    - Teşhis öncesi değerlendirme desteği
    - Şüpheli vakaların ön analizi
    - İkinci görüş alma aracı
    - Eğitim ve araştırma materyali
    
    #### 👨‍🎓 **Akademik Araştırma**
    - Makine öğrenmesi algoritma karşılaştırması
    - Yeni model geliştirme ve test etme
    - Özellik mühendisliği çalışmaları
    - Performans analizi ve optimizasyon
    """)

with scenario_col2:
    st.markdown("""
    #### 🔬 **Veri Bilimciler İçin**
    - Model prototipleme ve geliştirme
    - Farklı preprocessing teknikleri deneme
    - Hyperparameter tuning çalışmaları
    - Ensemble model geliştirme
    
    #### 📚 **Eğitim Amaçlı**
    - Makine öğrenmesi kavramları öğretimi
    - Pratik uygulama deneyimi
    - Model değerlendirme teknikleri
    - Gerçek dünya problemi simülasyonu
    """)

# Navigasyon rehberi
st.subheader("🗺️ Platform Navigasyon Rehberi")

st.markdown("""
Platformun tüm özelliklerinden yararlanmak için aşağıdaki sırayı takip edebilirsiniz:

| Sayfa | Açıklama |
|-------|----------|
| **🏠 Ana Sayfa** | Platform detayları ve teknik bilgiler |
| **📊 Veri Analizi** | Veri setini keşfedin, dağılımları inceleyin |
| **🧠 Model Eğitimi** | Kendi modelinizi eğitin ve optimize edin |
| **🔍 Hazır Modeller** | Önceden eğitilmiş modelleri karşılaştırın |
| **🔮 Tahmin** | Gerçek verilerle tahmin yapın |
| **⚖️ Özellik Önemi** | Modellerin hangi özelliklere odaklandığını görün |
""")

# Teknik altyapı
st.subheader("💻 Teknik Altyapı")

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.markdown("""
    #### **Makine Öğrenmesi Kütüphaneleri**
    - **Scikit-learn**: Klasik ML algoritmaları
    - **TensorFlow/Keras**: Derin öğrenme modelleri
    - **XGBoost**: Gradient boosting optimizasyonu
    - **Joblib**: Model serileştirme ve kaydetme
    
    #### **Veri İşleme**
    - **Pandas**: Veri manipülasyonu ve analizi
    - **NumPy**: Sayısal hesaplamalar
    - **Scikit-learn**: Preprocessing ve scaler'lar
    """)

with tech_col2:
    st.markdown("""
    #### **Görselleştirme**
    - **Matplotlib**: Statik grafikler
    - **Seaborn**: İstatistiksel görselleştirmeler
    - **Plotly**: Etkileşimli grafikler
    - **Streamlit**: Web arayüzü
    
    #### **Model Değerlendirme**
    - **ROC-AUC**: Model performans metrikleri
    - **Confusion Matrix**: Sınıflandırma analizi
    - **Feature Importance**: Özellik önemi analizi
    """)

# Güvenlik ve sorumluluk
st.warning("""
⚠️ **Önemli Uyarı**: Bu platform sadece eğitim, araştırma ve karar destek amaçlıdır. 
Kesinlikle profesyonel tıbbi teşhis, tedavi planlaması veya hasta bakımı kararları yerine geçmez. 
Tüm tıbbi kararlar uzman hekim gözetiminde alınmalıdır.
""")

# Başlangıç önerileri
st.info("""
🚀 **Hızlı Başlangıç**: Platform yeniyseniz, önce **Ana Sayfa** bölümünü ziyaret ederek detaylı bilgi alın, 
ardından **Veri Analizi** sayfasından başlayarak örnek veri setini inceleyin!
""")

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
Metin Ilgar Mutlu
</div>
""", unsafe_allow_html=True)
