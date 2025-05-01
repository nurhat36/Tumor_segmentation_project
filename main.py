# Örnek bir U-Net segmentasyon projesi iskeleti için gerekli dosya yapısını hazırlayalım
import os

base_path = "C:\\Users\\Nurhat\\OneDrive\\Masaüstü\\python\\data projesi\\tumor_segmentation_project"

# Klasör yapısını oluştur
folders = [
    "images",        # MR görüntüleri
    "masks",         # Maskeler (etiketli görüntüler)
    "models",        # Eğitilmiş modellerin saklandığı yer
    "notebooks",     # Eğitim ve test işlemleri için Jupyter dosyaları
    "scripts",       # Python script dosyaları
    "outputs"        # Sonuçların kaydedileceği yer
]

for folder in folders:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)

base_path  # Geriye temel proje klasörü yolu döndürülür.
