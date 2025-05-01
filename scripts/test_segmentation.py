import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model


def load_image(image_path, size=(128, 128)):  # Modelin beklediği boyut 128x128 olarak ayarlandı
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    img = cv2.resize(img, size) / 255.0
    return img.reshape(1, size[0], size[1], 1)


def process_and_save_mask(predicted_mask, original_image, folder="masks"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    mask = predicted_mask[0, :, :, 0]
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Dinamik eşikleme
    thresh_val = max(np.mean(mask_uint8) + 20, 150)
    _, binary_mask = cv2.threshold(mask_uint8, thresh_val, 255, cv2.THRESH_BINARY)

    # Görüntüyü orijinal boyutuna yeniden boyutlandır
    binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

    # Kontur bulma
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Orijinal görüntüyü renkli hale getir
    if len(original_image.shape) == 2:
        original_image_colored = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        original_image_colored = original_image.copy()

    # Konturları çiz
    cv2.drawContours(original_image_colored, contours, -1, (0, 255, 0), 5)

    # Görselleştirme
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Tahmin Edilen Maske")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Konturlu Sonuç")
    plt.imshow(cv2.cvtColor(original_image_colored, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(folder, f"contour_{timestamp}.png")
    cv2.imwrite(save_path, original_image_colored)
    print(f"Sonuç kaydedildi: {save_path}")


def main():
    model_path = '../models/unet_model.h5'
    test_image_path = '../images/images (9).jpg'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test görüntüsü bulunamadı: {test_image_path}")

    model = load_model(model_path)

    # Orijinal görüntüyü yükle (gösterim için)
    original_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    original_image_display = original_image.copy()

    # Model için giriş hazırla (128x128)
    image = load_image(test_image_path, size=(128, 128))
    prediction = model.predict(image)

    # İşleme fonksiyonunu çağır
    process_and_save_mask(prediction, original_image_display)


if __name__ == "__main__":
    main()