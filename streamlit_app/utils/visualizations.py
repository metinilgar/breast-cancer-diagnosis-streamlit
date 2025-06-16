import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Matplotlib stil ayarları
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.1)
sns.set_style("whitegrid")

def plot_histograms(df, numerical_cols=None, figsize=(15, 10)):
    """
    Sayısal sütunların histogramlarını çizen fonksiyon
    
    Args:
        df (DataFrame): Veri seti
        numerical_cols (list, optional): Çizilecek sayısal sütunlar
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Veri setindeki sayısal sütunları seç
    df_numeric = df[numerical_cols]
    
    # Histogramları çiz
    fig, axes = plt.subplots(len(df_numeric.columns) // 3 + 1, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(df_numeric.columns):
        if 'target' in col.lower() or 'class' in col.lower():
            sns.countplot(x=col, data=df, ax=axes[i])
        else:
            sns.histplot(df_numeric[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Dağılımı')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Boş alt grafikleri gizle
    for i in range(len(df_numeric.columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Korelasyon matrisini ısı haritası olarak çizen fonksiyon
    
    Args:
        df (DataFrame): Veri seti
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    # Sayısal sütunları seç
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    
    # Korelasyon matrisini hesapla
    corr = df_numeric.corr()
    
    # Isı haritasını çiz
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .8})
    
    plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt.gcf()

def plot_pairplot(df, target_col='target', sample_n=None, features=None):
    """
    Seçili özelliklerin çiftli grafik görselleştirmesi
    
    Args:
        df (DataFrame): Veri seti
        target_col (str, optional): Hedef sütun adı
        sample_n (int, optional): Örnek sayısı (büyük veri setleri için)
        features (list, optional): Görselleştirilecek özellikler
        
    Returns:
        Figure: Seaborn pairplot figürü
    """
    # Büyük veri setleri için örnekleme
    if sample_n is not None and len(df) > sample_n:
        df_sample = df.sample(sample_n, random_state=42)
    else:
        df_sample = df
    
    # Görselleştirilecek özellikleri seç
    if features is None:
        # En önemli 5 özelliği seç (veya daha az, eğer 5'ten az özellik varsa)
        num_features = min(5, len(df.columns) - 1)
        features = list(df.select_dtypes(include=['float64', 'int64']).columns[:num_features])
    
    if target_col not in features and target_col in df.columns:
        features.append(target_col)
    
    # Pairplot oluştur
    pair_plot = sns.pairplot(df_sample[features], hue=target_col if target_col in df.columns else None,
                             diag_kind='kde', plot_kws={'alpha': 0.6}, height=2.5)
    
    pair_plot.fig.suptitle('Çiftli Özellik Görselleştirmesi', y=1.02, fontsize=16)
    plt.tight_layout()
    
    return pair_plot.fig

def plot_feature_distribution(df, feature_col, target_col='target', figsize=(12, 6)):
    """
    Bir özelliğin hedef sınıfa göre dağılımını çizen fonksiyon
    
    Args:
        df (DataFrame): Veri seti
        feature_col (str): Özellik sütunu adı
        target_col (str, optional): Hedef sütun adı
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    plt.figure(figsize=figsize)
    
    target_values = df[target_col].unique()
    
    for target_value in target_values:
        sns.kdeplot(df[df[target_col] == target_value][feature_col], 
                   label=f"Sınıf {target_value}")
    
    plt.title(f'{feature_col} Özelliğinin Sınıflara Göre Dağılımı', fontsize=14)
    plt.xlabel(feature_col, fontsize=12)
    plt.ylabel('Yoğunluk', fontsize=12)
    plt.legend(title=target_col)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_boxplots(df, numerical_cols=None, target_col='target', figsize=(15, 10)):
    """
    Sayısal özelliklerin kutu grafiklerini çizen fonksiyon
    
    Args:
        df (DataFrame): Veri seti
        numerical_cols (list, optional): Sayısal sütunlar listesi
        target_col (str, optional): Hedef sütun adı
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
    
    # Kutu grafiklerini çiz
    fig, axes = plt.subplots(len(numerical_cols) // 3 + 1, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} Kutu Grafiği')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Boş alt grafikleri gizle
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_df, top_n=15, figsize=(12, 8)):
    """
    Özellik önemini çizen fonksiyon
    
    Args:
        importance_df (DataFrame): Özellik önemi dataframe'i ('feature' ve 'importance' sütunları)
        top_n (int, optional): Gösterilecek üst özellik sayısı
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    # Özellik sayısını kontrol et
    n_features = min(top_n, len(importance_df))
    
    # Top N özelliklerini seç
    top_features = importance_df.nlargest(n_features, 'importance')
    
    plt.figure(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'En Önemli {n_features} Özellik', fontsize=16)
    plt.xlabel('Önem Derecesi', fontsize=14)
    plt.ylabel('Özellik', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_pca_visualization(X, y, n_components=2, figsize=(10, 8)):
    """
    PCA kullanarak veri setini 2D veya 3D olarak görselleştiren fonksiyon
    
    Args:
        X (array): Özellik matrisi
        y (array): Hedef değişken
        n_components (int, optional): PCA bileşen sayısı (2 veya 3)
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    # PCA uygula
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Varyans oranını hesapla
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # 2D görselleştirme
    if n_components == 2:
        plt.figure(figsize=figsize)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='w', linewidth=0.5)
        
        plt.xlabel(f'Birinci Bileşen ({explained_variance[0]:.2f}%)', fontsize=12)
        plt.ylabel(f'İkinci Bileşen ({explained_variance[1]:.2f}%)', fontsize=12)
        plt.title('PCA 2D Görselleştirme', fontsize=14)
        
        # Renk çubuğu ekle
        plt.colorbar(scatter, label='Hedef Sınıf')
        
    # 3D görselleştirme
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.8, edgecolors='w', linewidth=0.5)
        
        ax.set_xlabel(f'Birinci Bileşen ({explained_variance[0]:.2f}%)', fontsize=12)
        ax.set_ylabel(f'İkinci Bileşen ({explained_variance[1]:.2f}%)', fontsize=12)
        ax.set_zlabel(f'Üçüncü Bileşen ({explained_variance[2]:.2f}%)', fontsize=12)
        ax.set_title('PCA 3D Görselleştirme', fontsize=14)
        
        # Renk çubuğu ekle
        fig.colorbar(scatter, ax=ax, label='Hedef Sınıf')
    
    plt.tight_layout()
    return plt.gcf()

def plot_tsne_visualization(X, y, figsize=(10, 8), perplexity=30, n_iter=300):
    """
    t-SNE kullanarak veri setini 2D olarak görselleştiren fonksiyon
    
    Args:
        X (array): Özellik matrisi
        y (array): Hedef değişken
        figsize (tuple, optional): Figür boyutu
        perplexity (int, optional): t-SNE perplexity parametresi
        n_iter (int, optional): t-SNE iterasyon sayısı
        
    Returns:
        Figure: Matplotlib figürü
    """
    # t-SNE uygula
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 2D görselleştirme
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='w', linewidth=0.5)
    
    plt.xlabel('t-SNE Boyut 1', fontsize=12)
    plt.ylabel('t-SNE Boyut 2', fontsize=12)
    plt.title('t-SNE 2D Görselleştirme', fontsize=14)
    
    # Renk çubuğu ekle
    plt.colorbar(scatter, label='Hedef Sınıf')
    plt.tight_layout()
    
    return plt.gcf()

def plot_roc_comparison(models_results, figsize=(12, 8)):
    """
    Birden fazla modelin ROC eğrilerini karşılaştıran fonksiyon
    
    Args:
        models_results (dict): Model adı: (fpr, tpr, auc) formatlı sözlük
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    plt.figure(figsize=figsize)
    
    for model_name, (fpr, tpr, roc_auc) in models_results.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (1 - Özgüllük)', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı (Duyarlılık)', fontsize=12)
    plt.title('Alıcı İşletim Karakteristiği (ROC) Eğrisi Karşılaştırması', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_learning_curve(history):
    """
    Yapay sinir ağı eğitim geçmişini görselleştiren fonksiyon
    
    Args:
        history (dict): tf.keras model.fit() geçmişi
        
    Returns:
        Figure: Matplotlib figürü
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Doğruluk grafiği
    axes[0].plot(history['accuracy'], label='Eğitim', marker='o')
    axes[0].plot(history['val_accuracy'], label='Doğrulama', marker='x')
    axes[0].set_title('Model Doğruluğu', fontsize=14)
    axes[0].set_ylabel('Doğruluk', fontsize=12)
    axes[0].set_xlabel('Dönem (Epoch)', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Kayıp grafiği
    axes[1].plot(history['loss'], label='Eğitim', marker='o')
    axes[1].plot(history['val_loss'], label='Doğrulama', marker='x')
    axes[1].set_title('Model Kaybı', fontsize=14)
    axes[1].set_ylabel('Kayıp', fontsize=12)
    axes[1].set_xlabel('Dönem (Epoch)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_metrics_comparison(models_metrics, metrics_list=None, figsize=(12, 8)):
    """
    Birden fazla modelin metriklerini karşılaştıran çubuk grafiği
    
    Args:
        models_metrics (dict): Model adı: metrikler sözlüğü formatlı sözlük
        metrics_list (list, optional): Gösterilecek metrikler listesi
        figsize (tuple, optional): Figür boyutu
        
    Returns:
        Figure: Matplotlib figürü
    """
    if metrics_list is None:
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # DataFrame oluştur
    comparison_data = []
    for model_name, metrics in models_metrics.items():
        row = {'model': model_name}
        for metric in metrics_list:
            if metric in metrics:
                row[metric] = metrics[metric]
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Çubuk grafiği çiz
    plt.figure(figsize=figsize)
    df_melted = pd.melt(df_comparison, id_vars=['model'], value_vars=metrics_list, 
                        var_name='Metrik', value_name='Değer')
    
    sns.barplot(x='model', y='Değer', hue='Metrik', data=df_melted)
    plt.title('Model Performans Karşılaştırması', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Metrik Değeri', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend(title='Metrik', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt.gcf()

def plot_interactive_scatter(df, x_col, y_col, color_col=None, size_col=None, title=None):
    """
    Plotly ile etkileşimli dağılım grafiği oluşturan fonksiyon
    
    Args:
        df (DataFrame): Veri seti
        x_col (str): X ekseni için sütun adı
        y_col (str): Y ekseni için sütun adı
        color_col (str, optional): Renk kodlaması için sütun adı
        size_col (str, optional): Boyut kodlaması için sütun adı
        title (str, optional): Grafik başlığı
        
    Returns:
        Figure: Plotly figürü
    """
    if title is None:
        title = f"{x_col} vs {y_col}"
    
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                    title=title, 
                    labels={x_col: x_col, y_col: y_col},
                    hover_data=df.columns)
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        legend=dict(
            title=color_col if color_col is not None else "",
            bgcolor='rgba(255, 255, 255, 0.8)',
        ),
        font=dict(size=12)
    )
    
    return fig

def plot_interactive_3d_scatter(df, x_col, y_col, z_col, color_col=None, title=None):
    """
    Plotly ile etkileşimli 3D dağılım grafiği oluşturan fonksiyon
    
    Args:
        df (DataFrame): Veri seti
        x_col (str): X ekseni için sütun adı
        y_col (str): Y ekseni için sütun adı
        z_col (str): Z ekseni için sütun adı
        color_col (str, optional): Renk kodlaması için sütun adı
        title (str, optional): Grafik başlığı
        
    Returns:
        Figure: Plotly figürü
    """
    if title is None:
        title = f"3D: {x_col} vs {y_col} vs {z_col}"
    
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                       title=title,
                       labels={x_col: x_col, y_col: y_col, z_col: z_col})
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            aspectmode='cube'
        ),
        font=dict(size=12)
    )
    
    return fig

def plot_interactive_feature_importance(importance_df, top_n=15, title="Özellik Önemi"):
    """
    Plotly ile etkileşimli özellik önemi grafiği oluşturan fonksiyon
    
    Args:
        importance_df (DataFrame): Özellik önemi dataframe'i ('feature' ve 'importance' sütunları)
        top_n (int, optional): Gösterilecek üst özellik sayısı
        title (str, optional): Grafik başlığı
        
    Returns:
        Figure: Plotly figürü
    """
    # Özellik sayısını kontrol et
    n_features = min(top_n, len(importance_df))
    
    # Top N özelliklerini seç ve büyükten küçüğe sırala
    top_features = importance_df.nlargest(n_features, 'importance').sort_values('importance')
    
    fig = px.bar(top_features, x='importance', y='feature', title=title,
                orientation='h',
                labels={'importance': 'Önem Derecesi', 'feature': 'Özellik'},
                color='importance',
                color_continuous_scale='Viridis')
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        font=dict(size=12)
    )
    
    return fig
