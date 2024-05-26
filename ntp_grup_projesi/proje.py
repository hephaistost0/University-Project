# Gerekli kütüphaneleri içe aktarır
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Veriyi CSV dosyasından okur
df = pd.read_csv('breast_cancer.csv')
data = pd.read_csv('breast_cancer.csv')

# Özellik isimlerini ve sınıf isimlerini belirler
feature_names = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
class_names = ['iyi huylu', 'kötü huylu']

# Veri kümesini ve sınıf etiketlerini belirler
X = df[feature_names].values

# K-means modelini oluşturur
kmeans = KMeans(n_clusters=2, random_state=0)

# Modeli veriye uygular
kmeans.fit(X)

# Veri noktalarını ve küme merkezlerini görselleştirir
plt.figure(figsize=(12, 8))

# Her bir özelliğin küme merkezlerinin ortalamasını sütun grafik olarak gösterir
x = np.arange(len(feature_names)) * 2  # x ekseninde özelliklerin konumlarını belirlemek için, her özelliğin arasında 2 birim boşluk bırakır
width = 0.35  # sütun genişliği

# Benign ve Malignant için sütun grafiklerini oluşturur
for cluster_label in range(2):
    cluster_means = [np.mean(X[kmeans.labels_ == cluster_label, i]) for i in range(len(feature_names))]
    plt.bar(x + cluster_label * width, cluster_means, width, label=class_names[cluster_label])

plt.xlabel('Özellik')
plt.ylabel('Ortalama Değer')
plt.title('K-means Kümeleme Sonuçları (Sütun Grafik)')
plt.xticks(x + width / 2, feature_names, rotation=45, ha='right')  # Özellik isimlerini 45 derece döndürerek daha iyi görünmesini sağlar
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

y = data['Class']

# Veriyi ölçeklendirir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means modeli oluşturur
kmns = KMeans(n_clusters=2, init='k-means++', random_state=42)

# Modeli eğitir ve tahminleri alır
kY = kmns.fit_predict(X_scaled)

# Veriyi 2D olarak görselleştirmek için ilk iki özelliği kullanacağız
# İki subplot oluşturur
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))

# K-Means kümeleri plotlar
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kY, cmap="jet", edgecolor="None", alpha=0.7)
ax1.set_title('K-Means Kümeleme Grafiği')

# Gerçek sınıfları plotlar
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="jet", edgecolor="None", alpha=0.7)
ax2.set_title('Gerçek Kümeleme Grafiği')
plt.show()

# Kümeler arasındaki uzaklıkları hesaplar
distances = kmeans.transform(X)

# Kümeler arasındaki farkı gösteren bir scatter plot oluşturur
plt.figure(figsize=(8, 6))
plt.scatter(distances[:, 0], distances[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Benign Uzaklık')
plt.ylabel('Malignant Uzaklık')
plt.title('Veri Noktalarının Kümeler Arasındaki Uzaklıkları')
plt.colorbar(label='Kümeler Etiketi')
plt.show()

# Gerçek sınıflarla tahmin edilen sınıfları karşılaştırır
predicted_labels = kmeans.labels_
true_labels = y.replace({'iyi huylu': 0, 'kötü huylu': 1})  # Sınıf etiketlerini sayısal değerlere dönüştürür

# Veriyi ölçeklendirir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Farklı küme sayılarını dener ve en iyi performansı veren küme sayısını seçer
best_accuracy = 0
best_k = 0

for k in range(2, 10):
    pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=0))
    pipeline.fit(X)
    labels = pipeline.predict(X)
    accuracy = accuracy_score(true_labels, labels)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# En iyi performansı veren küme sayısını kullanarak K-means modelini oluşturur
best_pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=best_k, random_state=0))
best_pipeline.fit(X)
best_labels = best_pipeline.predict(X)

# Gerçek sınıflarla tahmin edilen sınıfları karşılaştırır
accuracy = accuracy_score(true_labels, best_labels)

# Her bir veri noktasının gerçek sınıfı ve tahmin edilen küme etiketini gösteren bir scatter plot oluşturur
plt.figure(figsize=(8, 6))
plt.scatter(range(len(true_labels)), true_labels, c=best_labels, cmap='viridis')
plt.xlabel('Veri Noktası İndeksi')
plt.ylabel('Sınıf Etiketi')
plt.title(f'Gerçek ve Tahmin Edilen Sınıflar (Doğruluk: {accuracy:.2f})')
plt.colorbar(label='Tahmin Edilen Küme Etiketi')
plt.show()

# Skorları hesaplar ve yazdırır

f1 = f1_score(true_labels, best_labels, average='micro')  # F1 puanını hesaplar
precision = precision_score(true_labels, best_labels, average='micro')  # Hassasiyet (Precision) skorunu hesaplar
recall = recall_score(true_labels, best_labels, average='micro')  # Geri Çağırma (Recall) skorunu hesaplar

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Veriler
scores = [accuracy, f1, precision, recall]
labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

# Çubuk grafik oluşturma
plt.figure(figsize=(10, 5))
plt.bar(labels, scores, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Skor')
plt.title('Sınıflandırma Metrikleri')
plt.ylim(0, 1)  # Y eksenini 0 ile 1 arasında sınırla
plt.show()

