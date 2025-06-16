import numpy as np
import pandas as pd
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

def train_classical_model(X_train, y_train, model_type="random_forest", params=None):
    """
    Klasik makine öğrenmesi modeli eğitme fonksiyonu
    
    Args:
        X_train (array): Eğitim verileri özellikleri
        y_train (array): Eğitim verileri hedefleri
        model_type (str): Model türü
        params (dict, optional): Model parametreleri
        
    Returns:
        object: Eğitilmiş model nesnesi
    """
    default_params = {
        "random_forest": {"n_estimators": 100, "random_state": 42},
        "logistic_regression": {"C": 1.0, "random_state": 42, "max_iter": 1000},
        "svm": {"C": 1.0, "kernel": "rbf", "random_state": 42},
        "knn": {"n_neighbors": 5},
        "decision_tree": {"max_depth": 5, "random_state": 42},
        "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (100,), "max_iter": 300, "random_state": 42}
    }
    
    # Parametreleri birleştir (varsayılan + kullanıcı verilen)
    if params is None:
        params = {}
    model_params = {**default_params.get(model_type, {}), **params}
    
    # Model türüne göre sınıflandırıcı oluştur
    if model_type == "random_forest":
        model = RandomForestClassifier(**model_params)
    elif model_type == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_type == "svm":
        model = SVC(**model_params, probability=True)
    elif model_type == "knn":
        model = KNeighborsClassifier(**model_params)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(**model_params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**model_params)
    elif model_type == "mlp":
        model = MLPClassifier(**model_params)
    else:
        raise ValueError(f"Geçersiz model türü: {model_type}")
    
    # Modeli eğit
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    model_info = {
        "type": model_type,
        "params": model_params,
        "training_time": training_time
    }
    
    return model, model_info

def train_neural_network(X_train, y_train, input_shape, params=None):
    """
    TensorFlow/Keras ile yapay sinir ağı eğitme fonksiyonu
    
    Args:
        X_train (array): Eğitim verileri özellikleri
        y_train (array): Eğitim verileri hedefleri
        input_shape (int): Girdi boyutu
        params (dict, optional): Model parametreleri
        
    Returns:
        object: Eğitilmiş model ve model bilgileri
    """
    default_params = {
        "layers": [64, 32],
        "activation": "relu",
        "dropout_rate": 0.2,
        "optimizer": "adam",
        "epochs": 50,
        "batch_size": 32
    }
    
    # Parametreleri birleştir
    if params is None:
        params = {}
    model_params = {**default_params, **params}
    
    # Model oluştur
    model = Sequential()
    model.add(Dense(model_params["layers"][0], activation=model_params["activation"], input_shape=(input_shape,)))
    model.add(Dropout(model_params["dropout_rate"]))
    
    for units in model_params["layers"][1:]:
        model.add(Dense(units, activation=model_params["activation"]))
        model.add(Dropout(model_params["dropout_rate"]))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Modeli derle
    model.compile(optimizer=model_params["optimizer"],
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Erken durma mekanizması
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Modeli eğit
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=model_params["epochs"],
        batch_size=model_params["batch_size"],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    training_time = time.time() - start_time
    
    model_info = {
        "type": "neural_network",
        "params": model_params,
        "training_time": training_time,
        "history": history.history
    }
    
    return model, model_info

def evaluate_model(model, X_test, y_test, model_type="classical"):
    """
    Model performansını değerlendiren fonksiyon
    
    Args:
        model (object): Değerlendirilecek model
        X_test (array): Test verileri özellikleri
        y_test (array): Test verileri hedefleri
        model_type (str): Model türü (classical veya neural_network)
        
    Returns:
        dict: Değerlendirme metrikleri
    """
    # Tahminleri al
    if model_type == "classical":
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            # SVC gibi modeller için decision_function kullan
            if hasattr(model, "decision_function"):
                y_pred_proba = model.decision_function(X_test)
            else:
                y_pred_proba = y_pred
    else:  # neural_network
        y_pred_proba = model.predict(X_test).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = None
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics

def save_model(model, model_info, metrics, file_name, directory="models/custom", preprocessing_info=None, scaler=None):
    """
    Modeli kaydetme fonksiyonu
    
    Args:
        model (object): Kaydedilecek model
        model_info (dict): Model bilgileri
        metrics (dict): Model metrikleri
        file_name (str): Kaydedilecek dosya adı
        directory (str, optional): Kaydedilecek dizin
        preprocessing_info (dict, optional): Ön işleme bilgileri (columns_to_drop, feature_names vb.)
        scaler (object, optional): Eğitimde kullanılan scaler
        
    Returns:
        str: Kaydedilen dosya yolu
    """
    os.makedirs(directory, exist_ok=True)
    
    # Modelin türünü kontrol et
    if model_info["type"] == "neural_network":
        model_path = os.path.join(directory, f"{file_name}.h5")
        model.save(model_path)
    else:
        model_path = os.path.join(directory, f"{file_name}.joblib")
        joblib.dump(model, model_path)
    
    # Scaler'ı ayrı bir dosyaya kaydet
    if scaler is not None:
        scaler_path = os.path.join(directory, f"{file_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
    
    # Model bilgilerini ve metrikleri kaydet
    info_path = os.path.join(directory, f"{file_name}_info.joblib")
    save_info = {
        "model_info": model_info,
        "metrics": metrics,
        "preprocessing_info": preprocessing_info or {},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "has_scaler": scaler is not None
    }
    joblib.dump(save_info, info_path)
    
    return model_path

def load_model(file_name, directory="models/custom"):
    """
    Modeli yükleme fonksiyonu
    
    Args:
        file_name (str): Yüklenecek dosya adı
        directory (str, optional): Yüklenecek dizin
        
    Returns:
        tuple: (model, model_info, metrics, preprocessing_info, scaler)
    """
    # Önce bilgileri yükle
    info_path = os.path.join(directory, f"{file_name}_info.joblib")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Model bilgi dosyası bulunamadı: {info_path}")
    
    model_data = joblib.load(info_path)
    model_info = model_data["model_info"]
    metrics = model_data["metrics"]
    preprocessing_info = model_data.get("preprocessing_info", {})
    
    # Scaler'ı yükle (eğer varsa)
    scaler = None
    if model_data.get("has_scaler", False):
        scaler_path = os.path.join(directory, f"{file_name}_scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
    
    # Modelin türüne göre yükleme işlemi
    if model_info["type"] == "neural_network":
        model_path = os.path.join(directory, f"{file_name}.h5")
        model = tf.keras.models.load_model(model_path)
    else:
        model_path = os.path.join(directory, f"{file_name}.joblib")
        model = joblib.load(model_path)
    
    return model, model_info, metrics, preprocessing_info, scaler

def list_available_models(directory="models"):
    """
    Mevcut modelleri listeleyen fonksiyon
    
    Args:
        directory (str, optional): Aranacak dizin
        
    Returns:
        dict: Mevcut modeller bilgisi
    """
    models = {"ready": [], "custom": []}
    
    # Hazır modelleri ara
    ready_dir = os.path.join(directory, "ready")
    custom_dir = os.path.join(directory, "custom")
    
    for dir_name, model_type in [(ready_dir, "ready"), (custom_dir, "custom")]:
        if os.path.exists(dir_name):
            # joblib dosyalarını ara
            for file in os.listdir(dir_name):
                if file.endswith("_info.joblib"):
                    model_name = file.replace("_info.joblib", "")
                    try:
                        # Model bilgilerini yükle
                        info_path = os.path.join(dir_name, file)
                        model_data = joblib.load(info_path)
                        model_info = model_data["model_info"]
                        metrics = model_data["metrics"]
                        
                        models[model_type].append({
                            "name": model_name,
                            "type": model_info["type"],
                            "accuracy": metrics["accuracy"],
                            "f1_score": metrics["f1_score"],
                            "saved_at": model_data.get("saved_at", "Unknown")
                        })
                    except:
                        # Yükleme hatası durumunda bu modeli atla
                        pass
    
    return models

def plot_confusion_matrix(cm, class_names=None):
    """
    Karmaşıklık matrisini görselleştiren fonksiyon
    
    Args:
        cm (array): Karmaşıklık matrisi
        class_names (list, optional): Sınıf isimleri
        
    Returns:
        Figure: Matplotlib figürü
    """
    if class_names is None:
        class_names = ["Benign", "Malignant"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.title('Karmaşıklık Matrisi')
    return plt.gcf()

def plot_roc_curve(y_test, y_pred_proba):
    """
    ROC eğrisini görselleştiren fonksiyon
    
    Args:
        y_test (array): Gerçek etiketler
        y_pred_proba (array): Tahmin olasılıkları
        
    Returns:
        Figure: Matplotlib figürü
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı')
    plt.ylabel('Doğru Pozitif Oranı')
    plt.title('Alıcı İşletim Karakteristiği (ROC) Eğrisi')
    plt.legend(loc="lower right")
    return plt.gcf()

def calculate_feature_importance(model, feature_names, model_type="classical"):
    """
    Özellik önemini hesaplayan fonksiyon
    
    Args:
        model (object): Eğitilmiş model
        feature_names (list): Özellik isimleri
        model_type (str): Model türü
        
    Returns:
        DataFrame: Özellik önem sıralaması
    """
    importances = None
    
    if model_type == "neural_network":
        # Sinir ağları için özellik önemi hesaplanamaz
        return pd.DataFrame({"feature": feature_names, "importance": [None] * len(feature_names)})
    
    if hasattr(model, "feature_importances_"):
        # Random Forest, Decision Tree gibi modeller için
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression, SVM gibi modeller için
        importances = np.abs(model.coef_[0])
    else:
        # Özellik önemi sağlamayan modeller için
        return pd.DataFrame({"feature": feature_names, "importance": [None] * len(feature_names)})
    
    # Özellik önemini DataFrame'e dönüştür
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    
    # Öneme göre sırala
    feature_importance = feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)
    
    return feature_importance

def apply_model_preprocessing(X, preprocessing_info, scaler=None):
    """
    Model yüklendiğinde aynı ön işleme adımlarını uygulayan fonksiyon
    
    Args:
        X (DataFrame): İşlenecek veri
        preprocessing_info (dict): Model ön işleme bilgileri
        scaler (object, optional): Model scaler'ı
        
    Returns:
        numpy.ndarray: İşlenmiş veri
    """
    if not preprocessing_info:
        # Eğer preprocessing bilgisi yoksa veriyi olduğu gibi döndür
        return X.values if isinstance(X, pd.DataFrame) else X
    
    # DataFrame kopyası oluştur
    X_processed = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    
    # Çıkarılacak sütunları uygula
    columns_to_drop = preprocessing_info.get("columns_to_drop", [])
    if columns_to_drop:
        # Sadece mevcut sütunları çıkar
        available_columns_to_drop = [col for col in columns_to_drop if col in X_processed.columns]
        if available_columns_to_drop:
            X_processed = X_processed.drop(columns=available_columns_to_drop)
    
    # Model özellik isimlerini kontrol et
    expected_features = preprocessing_info.get("feature_names", None)
    if expected_features:
        # Eğer beklenen özellikler belirtilmişse, sadece o sütunları kullan
        available_features = [col for col in expected_features if col in X_processed.columns]
        if available_features:
            X_processed = X_processed[available_features]
    
    # Scaler uygula (eğer varsa)
    if scaler is not None:
        try:
            X_processed = scaler.transform(X_processed)
        except Exception as e:
            # Scaler uygulanamıyorsa hata vermeden devam et
            print(f"Scaler uygulanamadı: {e}")
            X_processed = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed
    else:
        X_processed = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed
    
    return X_processed
