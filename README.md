# 🩺 Yapay Zeka Destekli Meme Kanseri Teşhis Platformu

**Geliştirici:** Metin Ilgar Mutlu

## 📋 Proje Özeti

Bu proje, meme kanseri teşhis süreçlerinde sağlık profesyonellerine, araştırmacılara ve veri bilimcilere kapsamlı destek sağlamak amacıyla geliştirilmiş interaktif bir yapay zeka platformudur. Wisconsin Breast Cancer veri seti üzerinde çalışan platform, 8'den fazla farklı makine öğrenmesi algoritmasını ve modern derin öğrenme tekniklerini bir araya getirerek, kullanıcıların hem hazır modelleri incelemesini hem de kendi özel modellerini geliştirmesini sağlar.

### 🎯 Platform Misyonu
- Tıbbi veri analizi süreçlerini hızlandırmak
- Makine öğrenmesi eğitimi için interaktif platform sağlamak
- Araştırma ve geliştirme süreçlerini desteklemek
- Yapay zeka teknolojilerinin tıp alanındaki potansiyelini göstermek

## 🏗️ Proje Mimarisi

```
BreastCancer_Metin-Ilgar-Mutlu/
├── notebooks/
│   └── Breast_Cancer.ipynb          # Detaylı veri analizi notebook'u
├── streamlit_app/
│   ├── app.py                       # Ana Streamlit uygulaması
│   ├── pages/
│   │   ├── 1_HomePage.py            # Detaylı platform rehberi
│   │   ├── 2_DataAnalysis.py        # Kapsamlı veri analizi modülü
│   │   ├── 3_TrainYourModel.py      # Model eğitimi modülü
│   │   ├── 4_ExploreReadyModel.py   # Hazır model karşılaştırması
│   │   ├── 5_Prediction.py          # Gerçek zamanlı tahmin modülü
│   │   └── 6_FeatureImportance.py   # Özellik önemi analizi
│   ├── utils/
│   │   ├── model_utils.py           # Model eğitimi ve değerlendirme
│   │   ├── preprocessing.py         # Veri ön işleme fonksiyonları
│   │   └── visualizations.py        # Görselleştirme araçları
│   ├── models/
│   │   ├── ready/                   # Önceden eğitilmiş modeller
│   │   └── custom/                  # Kullanıcı modelleri
│   ├── data/
│   │   └── data.csv                 # Wisconsin Breast Cancer veri seti
│   ├── requirements.txt             # Python bağımlılıkları
│   └── environment.yml              # Conda ortam dosyası
├── data/
│   └── data.csv                     # Ana veri seti
└── Rapor - Metin Ilgar Mutlu.pdf    # Proje raporu
```

## ✨ Platform Özellikleri

### 📊 Kapsamlı Veri Analizi Modülü
- **Veri Kalitesi Kontrolü**: Eksik veri tespiti, aykırı değer analizi
- **İstatistiksel Özetler**: Tanımlayıcı istatistikler, dağılım analizi
- **Gelişmiş Görselleştirmeler**: 
  - Histogram ve KDE dağılım grafikleri
  - Korelasyon ısı haritası
  - Çiftli özellik grafikleri (Pairplot)
  - Kutu grafikleri (sınıflara göre)
- **Boyut Azaltma ve Keşif**:
  - PCA analizi (2D/3D görselleştirme)
  - t-SNE görselleştirmesi
  - İnteraktif 3D grafikler (Plotly)
- **Aykırı Değer Tespiti**: IQR ve Z-Score yöntemleri

### 🤖 Gelişmiş Model Geliştirme Ortamı

#### Klasik Makine Öğrenmesi Algoritmaları
- 🌳 **Random Forest**: Ensemble learning, feature importance
- 📈 **Logistic Regression**: Linear model, probabilistic output
- 🎯 **Support Vector Machine (SVM)**: Kernel tricks, margin optimization
- 👥 **K-Nearest Neighbors (KNN)**: Instance-based learning
- 🌿 **Decision Tree**: Interpretable tree-based models
- 🚀 **Gradient Boosting**: XGBoost style boosting algorithms
- 🧠 **Multi-Layer Perceptron**: Classical neural network

#### Derin Öğrenme Modelleri
- 🔬 **Custom Neural Networks**: TensorFlow/Keras tabanlı
- ⚙️ **Flexible Architecture**: 1-5 katman arası özelleştirme
- 🎛️ **Hyperparameter Tuning**: Batch size, learning rate, dropout
- 📊 **Training Monitoring**: Real-time loss ve accuracy tracking

#### Veri Ön İşleme Seçenekleri
- 📏 **StandardScaler**: Z-score normalizasyonu
- 📐 **MinMaxScaler**: 0-1 arası ölçekleme
- 🚫 **NoScaler**: Ham veri ile çalışma
- 🗑️ **Feature Selection**: Manuel özellik seçimi ve çıkarma

### 🔬 Model Değerlendirme ve Tahmin Sistemi

#### Performans Metrikleri
- ✅ **Accuracy**: Genel doğruluk oranı
- 🎯 **Precision**: Pozitif tahmin kesinliği
- 📊 **Recall (Sensitivity)**: Gerçek pozitif yakalama oranı
- ⚖️ **F1-Score**: Precision ve recall harmonik ortalaması
- 📈 **ROC-AUC**: Receiver Operating Characteristic curve
- 🔢 **Confusion Matrix**: Detaylı sınıflandırma analizi

#### Model Karşılaştırma ve Analiz
- 📊 **Side-by-Side Comparison**: Çoklu model performans karşılaştırması
- 🎨 **Visual Analytics**: ROC curves, bar charts, heatmaps
- 📋 **Detailed Reports**: Comprehensive model evaluation reports
- 🏆 **Ranking System**: Performance-based model ranking
- 🌟 **Feature Importance**: Tree-based model feature rankings

## 💻 Teknik Altyapı

### Makine Öğrenmesi Kütüphaneleri
- **Scikit-learn**: Klasik ML algoritmaları
- **TensorFlow/Keras**: Derin öğrenme modelleri
- **XGBoost**: Gradient boosting optimizasyonu
- **Joblib**: Model serileştirme ve kaydetme

### Veri İşleme ve Görselleştirme
- **Pandas**: Veri manipülasyonu ve analizi
- **NumPy**: Sayısal hesaplamalar
- **Matplotlib/Seaborn**: Statik görselleştirmeler
- **Plotly**: Etkileşimli grafikler
- **Streamlit**: Web arayüzü

## 📚 Veri Kaynağı

**Wisconsin Breast Cancer Dataset**
- **Kaynak**: UCI Machine Learning Repository
- **Örneklem Boyutu**: 569 hasta verisi
- **Özellik Sayısı**: 30 sayısal özellik
- **Hedef Değişken**: Binary classification (Malignant/Benign)
- **Veri Kalitesi**: Eksik veri yok, yüksek kaliteli cleaned dataset

**Özellik Kategorileri:**
- **Radius**: Hücre çekirdeği çapı özellikleri
- **Texture**: Gri değer standart sapması
- **Perimeter**: Çekirdek çevresi
- **Area**: Çekirdek alanı
- **Smoothness**: Yarıçap uzunluklarındaki lokal değişim
- **Compactness**: (perimeter² / area - 1.0)
- **Concavity**: Kontürün konkav kısımlarının şiddeti
- **Symmetry**: Simetri özellikleri
- **Fractal Dimension**: Kıyı şeridi yaklaşımı - 1

## 🚀 Kurulum ve Çalıştırma

### Sistem Gereksinimleri
- Python 3.10

### 1. Repository'yi Klonlayın
```bash
git clone <repository-url>
cd BreastCancer_Metin-Ilgar-Mutlu
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
# Python venv ile
python -m venv breast_cancer_env
source breast_cancer_env/bin/activate  # Linux/Mac
# veya
breast_cancer_env\Scripts\activate     # Windows

# veya Conda ile
conda env create -f streamlit_app/environment.yml
conda activate breast-cancer-ml
```

### 3. Bağımlılıkları Yükleyin
```bash
cd streamlit_app
pip install -r requirements.txt
```

### 4. Uygulamayı Çalıştırın
```bash
streamlit run app.py
```

### 5. Tarayıcıda Açın
Uygulama otomatik olarak `http://localhost:8501` adresinde açılacaktır.

## 📖 Kullanım Kılavuzu

### Platform Navigasyonu
Platform 6 ana modülden oluşmaktadır:

| Sayfa | Açıklama | Ana Özellikler |
|-------|----------|----------------|
| **🏠 Ana Sayfa** | Platform detayları ve teknik bilgiler | Özellik listesi, teknik altyapı |
| **📊 Veri Analizi** | Veri setini keşfedin, dağılımları inceleyin | EDA, görselleştirme, PCA |
| **🧠 Model Eğitimi** | Kendi modelinizi eğitin ve optimize edin | 8+ algoritma, hyperparameter tuning |
| **🔍 Hazır Modeller** | Önceden eğitilmiş modelleri karşılaştırın | Model comparison, performance analysis |
| **🔮 Tahmin** | Gerçek verilerle tahmin yapın | Real-time prediction, batch processing |
| **⚖️ Özellik Önemi** | Modellerin hangi özelliklere odaklandığını görün | Feature importance, interpretability |

### Temel Kullanım Senaryoları

#### 1. Veri Analizi Yapma
```
Veri Analizi sayfası → Veri kaynağı seçin → Hedef sütunu belirleyin → 
Analiz modülünü seçin → Görselleştirmeleri inceleyin
```

#### 2. Model Eğitimi
```
Model Eğitimi sayfası → Veri ön işleme ayarları → Model türü seçin → 
Parametreleri ayarlayın → Model adı verin → Eğitimi başlatın
```

#### 3. Model Karşılaştırması
```
Hazır Modeller sayfası → Karşılaştırılacak modelleri seçin → 
Metrikleri inceleyin → ROC eğrileri ve confusion matrix'leri karşılaştırın
```

#### 4. Tahmin Yapma
```
Tahmin sayfası → Model seçin → Veri girişi yöntemi seçin → 
Özellikleri girin → Tahmin sonuçlarını görüntüleyin
```

## 🔧 Gelişmiş Özellikler

### Model Persistency
- Eğitilen modeller otomatik olarak `models/custom/` klasörüne kaydedilir
- Model metadata'sı (parametreler, metrikler, preprocessing bilgileri) ayrı dosyalarda saklanır
- Scaler bilgileri model ile birlikte kaydedilir

### Batch Processing
- Çoklu örnek tahminleri
- CSV dosyası yükleme ve işleme
- Sonuçları indirme

### Interactive Visualizations
- Plotly tabanlı 3D görselleştirmeler
- Zoom, pan, filter özellikleri
- Responsive tasarım

## ⚠️ Güvenlik ve Sorumluluk

**ÖNEMLİ UYARI**: Bu platform **sadece eğitim, araştırma ve metodoloji geliştirme** amaçlarına yöneliktir. 

**Kesinlikle aşağıdaki amaçlar için kullanılmamalıdır:**
- ❌ Klinik tanı ve teşhis kararları
- ❌ Tedavi planlaması ve hasta yönetimi
- ❌ Tıbbi gözetim ve screening
- ❌ Profesyonel tıbbi konsültasyon yerine geçecek kullanım

**Tıbbi Sorumluluk Reddi**: Tüm tıbbi kararlar lisanslı sağlık profesyonelleri tarafından, uygun klinik değerlendirme ve standart teşhis protokolleri çerçevesinde alınmalıdır.


## 📊 Performans Metrikleri

Platform, aşağıdaki standart ML metriklerini destekler:
- **Accuracy**: Genel doğruluk
- **Precision**: Pozitif prediktif değer
- **Recall**: Sensitivity/True Positive Rate
- **F1-Score**: Precision ve Recall harmonik ortalaması
- **ROC-AUC**: Area Under ROC Curve
- **Confusion Matrix**: Detaylı sınıflandırma matrisi

## 📞 Destek ve İletişim

- **Geliştirici**: Metin Ilgar Mutlu
- **E-posta**: contact@ilgarmutlu.com
- **GitHub**: [github.com/metinilgar](https://github.com/metinilgar)

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla ayrıntı için [LICENSE](https://github.com/metinilgar/breast-cancer-diagnosis-streamlit/blob/main/LICENSE) dosyasına bakın.

---
