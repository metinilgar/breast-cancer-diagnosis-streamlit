import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path=None):
    """
    Veri yükleme fonksiyonu
    Dosya yolu verilmezse varsayılan breast cancer veri seti yüklenir
    
    Args:
        file_path (str, optional): Yüklenecek veri dosyasının yolu
        
    Returns:
        DataFrame: Yüklenen veri seti
    """
    if file_path is None or not os.path.exists(file_path):
        # Varsayılan breast cancer veri setini yükle
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    # Dosya uzantısına göre yükleme işlemi
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Desteklenmeyen dosya formatı. CSV veya Excel dosyaları kullanın.")

def preprocess_data(df, target_column=None, test_size=0.2, random_state=42, columns_to_drop=None, scaler_type="StandardScaler"):
    """
    Veri ön işleme fonksiyonu
    
    Args:
        df (DataFrame): İşlenecek veri seti
        target_column (str, optional): Hedef sütun adı
        test_size (float, optional): Test veri seti boyutu
        random_state (int, optional): Rastgele tohum değeri
        columns_to_drop (list, optional): Eğitimden çıkarılacak sütun listesi
        scaler_type (str, optional): Kullanılacak scaler türü
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, warnings)
    """
    warnings = {
        "target_not_selected": False,
        "target_protected_from_deletion": False,
        "target_column_used": None
    }
    
    # Hedef sütunun seçilip seçilmediğini kontrol et
    if target_column is None or target_column == "":
        warnings["target_not_selected"] = True
        # Varsayılan olarak 'target' sütununu ara
        if 'target' in df.columns:
            target_column = 'target'
            warnings["target_column_used"] = target_column
        else:
            # Son sütunu hedef olarak kullan
            target_column = df.columns[-1]
            warnings["target_column_used"] = target_column
    
    # Başlangıçta veri setinin bir kopyasını oluştur
    df_processed = df.copy()
    
    # Kullanıcının seçtiği sütunları sil
    if columns_to_drop and len(columns_to_drop) > 0:
        # Hedef sütun silinmemeli
        if target_column in columns_to_drop:
            columns_to_drop.remove(target_column)
            warnings["target_protected_from_deletion"] = True
            print(f"Hedef sütun '{target_column}' silme işlemi dışında tutuldu.")
        
        print(f"Kullanıcı tarafından seçilen sütunlar siliniyor: {columns_to_drop}")
        df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Eksik değerlerin %40'ından fazlası boş olan sütunları sil
    missing_percentage = df_processed.isnull().mean()
    auto_columns_to_drop = missing_percentage[missing_percentage > 0.4].index.tolist()
    
    if auto_columns_to_drop:
        print(f"Değerlerinin %40'ından fazlası boş olan sütunlar siliniyor: {auto_columns_to_drop}")
        # Hedef sütun silinmemeli
        if target_column in auto_columns_to_drop:
            auto_columns_to_drop.remove(target_column)
            warnings["target_protected_from_deletion"] = True
            print(f"Hedef sütun '{target_column}' silinmedi.")
        
        df_processed = df_processed.drop(columns=auto_columns_to_drop)
    
    print(df_processed.select_dtypes(include=['float64', 'int64']).shape)
    print(len(df_processed.select_dtypes(include=['float64', 'int64']).columns))
    # Eksik değerleri doldur
    imputer = SimpleImputer(strategy='mean')
    df_filled = pd.DataFrame(imputer.fit_transform(df_processed.select_dtypes(include=['float64', 'int64'])), 
                              columns=df_processed.select_dtypes(include=['float64', 'int64']).columns)
    
    # Kategorik sütunları işle
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != target_column:  # Hedef sütun değilse
            le = LabelEncoder()
            df_filled[col] = le.fit_transform(df_processed[col])
    
    # Hedef değişkeni ayır ve gerekirse dönüştür
    if target_column in df_processed.columns:
        y = df_processed[target_column]
        
        # Hedef değişken kategorik mi kontrol et (object, category veya az sayıda benzersiz değer)
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y) or len(y.unique()) <= 5:
            # Kategorik değerleri 0 ve 1'e dönüştür
            le = LabelEncoder()
            y = le.fit_transform(y)  # Bu zaten numpy array döndürür
            print(f"Hedef sütun '{target_column}' kategorik olduğu için dönüştürüldü. Eşleşmeler: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        elif isinstance(y, pd.Series):
            # Eğer y hala pandas Series ise NumPy array'e dönüştür
            y = y.values
        
        # X veri setini hazırla
        X = df_filled.drop(target_column, axis=1) if target_column in df_filled.columns else df_filled
    else:
        raise ValueError(f"Hedef sütun '{target_column}' veri setinde bulunamadı.")
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Kullanıcının seçtiği scaler türüne göre scaler oluştur
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "NoScaler":
        # Ölçeklendirme yapmaz, sadece dummy scaler döndürür
        class NoScaler:
            def fit(self, X):
                return self
            def transform(self, X):
                return X
            def fit_transform(self, X):
                return X
        scaler = NoScaler()
    else:
        raise ValueError(f"Desteklenmeyen scaler türü: {scaler_type}")
    
    # Özellikleri ölçeklendir
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, warnings

def check_data_quality(df):
    """
    Veri kalitesini kontrol eden fonksiyon
    
    Args:
        df (DataFrame): Kontrol edilecek veri seti
        
    Returns:
        dict: Veri kalitesi raporu
    """
    report = {
        "satir_sayisi": len(df),
        "sutun_sayisi": len(df.columns),
        "eksik_degerler": df.isnull().sum().to_dict(),
        "eksik_deger_yuzdesi": (df.isnull().sum() / len(df) * 100).to_dict(),
        "veri_tipleri": df.dtypes.astype(str).to_dict(),
        "kategorik_sutunlar": list(df.select_dtypes(include=['object', 'category']).columns),
        "sayisal_sutunlar": list(df.select_dtypes(include=['float64', 'int64']).columns),
        "essiz_degerler": {col: df[col].nunique() for col in df.columns}
    }
    
    return report

def detect_outliers(df, method='iqr', columns=None):
    """
    Aykırı değerleri tespit eden fonksiyon
    
    Args:
        df (DataFrame): İncelenecek veri seti
        method (str, optional): Kullanılacak yöntem ('iqr' veya 'zscore')
        columns (list, optional): İncelenecek sütunlar
        
    Returns:
        dict: Sütun bazında aykırı değer indeksleri
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            # IQR (Çeyrekler Arası Aralık) yöntemi
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outliers[col] = list(outlier_indices)
            
        elif method == 'zscore':
            # Z-score yöntemi
            from scipy import stats
            z_scores = stats.zscore(df[col])
            abs_z_scores = np.abs(z_scores)
            outlier_indices = df[abs_z_scores > 3].index
            outliers[col] = list(outlier_indices)
    
    return outliers
