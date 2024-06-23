import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Pickle dosyasını yükle
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

# Verileri ve etiketleri al
X = data['data']
y = data['labels']

# Verileri sabit bir boyuta getir
max_length = max(len(x) for x in X)
X = [x + [0] * (max_length - len(x)) for x in X]

print("Number of features in training data:", max_length)

# NumPy dizisine dönüştür
X = np.array(X)
y = np.array(y)

# Verileri eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest sınıflandırıcısını oluştur ve eğit
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test verileri üzerinde tahminleri gerçekleştir
y_pred = clf.predict(X_test)

# Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Performans metriklerini ekrana yazdır
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Confusion Matrix'i hesapla
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix'i görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Sınıflandırma raporunu görselleştir
class_names = np.unique(y)
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
plt.figure(figsize=(12, 6))
sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap='Blues')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.title('Classification Report')
plt.show()

# Modeli kaydet
with open('hand_gesture_model.pkl', 'wb') as f:
    pickle.dump(clf, f)