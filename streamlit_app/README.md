# 🩺 Yapay Zeka Destekli Meme Kanseri Teşhis Platformu

Bu proje, meme kanseri teşhisinde sağlık profesyonellerine destek sağlamak amacıyla geliştirilmiş kapsamlı bir yapay zeka platformudur. Modern makine öğrenmesi ve derin öğrenme teknolojilerini kullanarak, hücre görüntülerinden elde edilen özellikler üzerinden kanser hücrelerinin iyi huylu (benign) veya kötü huylu (malignant) olup olmadığını tahmin etmeye yardımcı olur.

## 🚀 Özellikler

- **📊 Kapsamlı Veri Analizi**: İstatistiksel analiz, görselleştirme ve korelasyon analizi
- **🤖 Gelişmiş Model Eğitimi**: 7+ farklı makine öğrenmesi algoritması desteği
- **🔍 Akıllı Model Analizi**: Model performans karşılaştırması ve metrik analizi
- **🔮 Gerçek Zamanlı Tahmin**: Eğitilmiş modellerle anlık tahmin yapma
- **⚖️ Özellik Önemi Analizi**: Modellerin hangi özelliklere odaklandığını görme

## 📋 Gereksinimler

- **Python**: 3.10


## 🛠️ Kurulum

### Seçenek 1: Conda ile Kurulum (Önerilen)

1. **Anaconda veya Miniconda'yı indirin ve kurun**:
   - [Anaconda](https://www.anaconda.com/products/distribution) (tam paket)
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (minimal kurulum)

2. **Terminal/Command Prompt'u açın**

3. **Proje dizinine gidin**:
   ```bash
   cd breast_cancer_project/streamlit_app
   ```

4. **Conda ortamını oluşturun**:
   ```bash
   conda env create -f environment.yml
   ```

5. **Ortamı aktif edin**:
   ```bash
   conda activate BaykarProje
   ```

### Seçenek 2: Pip ile Kurulum

1. **Python 3.10'un kurulu olduğundan emin olun**:
   ```bash
   python --version
   ```

2. **Proje dizinine gidin**:
   ```bash
   cd breast_cancer_project/streamlit_app
   ```

3. **Virtual environment oluşturun (önerilen)**:
   ```bash
   python -m venv venv
   ```

4. **Virtual environment'ı aktif edin**:
   
   **Windows için**:
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux için**:
   ```bash
   source venv/bin/activate
   ```

5. **Gerekli paketleri yükleyin**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Uygulamayı Çalıştırma

1. **Terminal/Command Prompt'da proje dizininde olduğunuzdan emin olun**

2. **Conda kullanıyorsanız ortamın aktif olduğunu kontrol edin**:
   ```bash
   conda activate BaykarProje
   ```

3. **Streamlit uygulamasını çalıştırın**:
   ```bash
   streamlit run app.py
   ```

4. **Tarayıcınızda otomatik olarak açılacak olan adrese gidin**:
   - Genellikle: `http://localhost:8501`

5. **Eğer otomatik açılmazsa, terminalde görünen URL'yi tarayıcınıza kopyalayın**

## 📱 Kullanım

Uygulama çalıştırıldıktan sonra şu sayfalar mevcuttur:

1. **🏠 Ana Sayfa**: Platform özellikleri ve teknik bilgiler
2. **📊 Veri Analizi**: Veri setini keşfetme ve görselleştirme
3. **🧠 Model Eğitimi**: Kendi modelinizi eğitme ve optimize etme
4. **🔍 Hazır Modeller**: Önceden eğitilmiş modelleri karşılaştırma
5. **🔮 Tahmin**: Gerçek verilerle tahmin yapma
6. **⚖️ Özellik Önemi**: Model analizi ve özellik değerlendirmesi

### 🚦 İlk Adımlar

1. **Ana Sayfa**'dan başlayarak platform özelliklerini öğrenin
2. **Veri Analizi** sayfasından örnek veri setini inceleyin
3. **Model Eğitimi** ile kendi modelinizi oluşturun
4. **Tahmin** sayfasında sonuçları test edin

## 📊 Teknik Detaylar

**Ana Teknolojiler**:
- **Streamlit**: Web arayüzü
- **TensorFlow/Keras**: Derin öğrenme
- **Scikit-learn**: Makine öğrenmesi
- **Pandas/NumPy**: Veri işleme
- **Matplotlib/Plotly**: Görselleştirme

**Desteklenen Algoritmalar**:
- Lojistik Regresyon
- Random Forest
- SVM (Support Vector Machine)
- Gradient Boosting (XGBoost)
- Yapay Sinir Ağları
- K-Nearest Neighbors
- Decision Tree

---

**Geliştiren**: Metin Ilgar Mutlu  
**Tarih**: 2025  
**Lisans**: Eğitim ve Araştırma Amaçlı 