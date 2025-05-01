import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model


# Görüntü yükleme ve ön işleme
def load_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size) / 255.0
    return img.reshape(1, size[0], size[1], 1)


# Maskeyi kaydet ve kontur çiz
def process_and_save_mask(predicted_mask, original_image, folder="masks"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    mask = predicted_mask[0, :, :, 0]
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Maskeyi eşikle binarize et
    _, thresh = cv2.threshold(mask_uint8, 128, 255, cv2.THRESH_BINARY)

    # Maskede açık gri ve beyaz tonlarını seç
    # Gri tonlar için belirli bir aralık kullanabiliriz, örneğin 200-255 arası
    light_gray_mask = cv2.inRange(mask_uint8, 200, 255)

    # Konturları bul
    contours, _ = cv2.findContours(light_gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konturları orijinal maske üzerine çiz
    contour_image = np.zeros_like(mask_uint8)

    # Maskedeki açık gri ve beyaz bölgelere yeşil renkli konturlar çiz
    contour_image_colored = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)  # Renkli görsel yapıyoruz
    cv2.drawContours(contour_image_colored, contours, -1, color=(0, 255, 0), thickness=2)  # Yeşil renkli konturlar

    # Maskeyi kaydet
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(folder, f"mask_contour_{timestamp}.png")
    cv2.imwrite(save_path, contour_image_colored)

    # Orijinal, tahmin maskesi ve konturlu maskeyi yan yana göster
    plt.figure(figsize=(15, 5))

    # Orijinal Görüntü
    plt.subplot(1, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    # Tahmin Edilen Maske
    plt.subplot(1, 3, 2)
    plt.title("Tahmin Edilen Maske")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    # Konturlu Maske
    plt.subplot(1, 3, 3)
    plt.title("Konturlu Maske")
    plt.imshow(contour_image_colored)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Konturlu maske kaydedildi: {save_path}")


# Modeli yükle ve çalıştır
model_path = '../models/unet_model.h5'
test_image_path = '../images/images (9).jpg'

if not os.path.exists(model_path):
    print("Model dosyası bulunamadı.")
else:
    model = load_model(model_path)

    # Orijinal Görüntüyü Yükle (grayscale formatta)
    original_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (128, 128)) / 255.0  # Boyutlandırma ve normalizasyon

    # Görüntü işleme ve tahmin
    image = load_image(test_image_path)
    prediction = model.predict(image)

    # Kontur işle ve kaydet
    process_and_save_mask(prediction, original_image)
