import os
import cv2
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR    = os.path.expanduser(path)
ROTATE_ANGLES_TOP = [90, 180, 270]
ROTATE_ANGLES = [30,45,60]
IMAGE_SIZE  = [224, 224]

LABEL_ORDERS = {"clay&gravel":0,
                "sand&clay":1,
                "sand&gravel":2
               }

def CripImage(image):
    return image[500:3500, 500:4500, :]

def CripTopImage(image):
    return image[324:3324, 1000:4000, :]

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

def preprocess_images(data_dir, image_size, label_orders, rotate_angles_top, rotate_angles):
    X = []  # Görüntü verileri
    y = []  # Etiketler

    for label_name, label in label_orders.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Klasör bulunamadı: {folder_path}")
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            try:
                # Görüntüyü yükle
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Görüntü yüklenemedi: {image_path}")
                    continue

                # Fotoğraf isimlendirmesine göre kırpma işlemi
                if "IMG_TOP" in image_name:
                    image = CripTopImage(image)
                    current_rotate_angles = rotate_angles_top
                else:
                    image = CripImage(image)
                    current_rotate_angles = rotate_angles

                # Görüntüyü yeniden boyutlandır
                image_resized = cv2.resize(image, tuple(image_size))

                # Orijinal görüntüyü ekle
                X.append(image_resized)
                y.append(label)

                # Döndürülmüş görüntüleri ekle (ilgili açı listesine göre)
                for angle in current_rotate_angles:
                    rotated_image = rotate_image(image_resized, angle)
                    X.append(rotated_image)
                    y.append(label)

            except Exception as e:
                print(f"Hata oluştu: {e}, Görüntü: {image_path}")

    return np.array(X), np.array(y)

# Veriyi işleme
X, y = preprocess_images(DATA_DIR, IMAGE_SIZE, LABEL_ORDERS, ROTATE_ANGLES_TOP, ROTATE_ANGLES)
print(f"Toplam veri boyutu: {X.shape}, Etiket boyutu: {y.shape}")
# Verileri düzleştir
X_flat = X.reshape(X.shape[0], -1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# SVM modelini oluştur
svm_model = SVC(kernel='linear', C = 10)
recall = []

# 10-fold cross-validation kullanarak doğruluk hesapla
kf = KFold(n_splits=10, shuffle=True)

recall, accuracy, std = [], [], []

for i in range(50):
    recall_scores = cross_val_score(svm_model, X_scaled, y, cv=kf, scoring='recall_macro')
    recall.append(round(recall_scores.mean()*100, 2))

    accuracy_scores = cross_val_score(svm_model, X_scaled, y, cv=kf, scoring='accuracy')
    accuracy.append(round(accuracy_scores.mean()*100, 2))
    std.append(round(accuracy_scores.std()*100, 2))

print("Recall:", np.mean(recall))
print("Accuracy:", np.mean(accuracy))
print("Std:", np.mean(std))
print(recall)
print(accuracy)
print(std)
