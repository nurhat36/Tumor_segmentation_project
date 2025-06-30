import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from skimage import exposure


# 1. Özel metrik fonksiyonları global olarak tanımla
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# IoU metriği için yeni fonksiyon
def iou_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / (union + K.epsilon())


# 2. Gelişmiş U-Net Modeli
class TumorSegmentationModel:
    def __init__(self, input_shape=(128, 128, 1)):  # Giriş boyutunu 128x128x1 olarak ayarla
        self.input_shape = input_shape
        self.model = self.build_advanced_unet()

    def build_advanced_unet(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Encoder Path
        def encoder_block(x, filters, dropout_rate=0.2):
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            p = layers.MaxPooling2D((2, 2))(x)
            return x, p

        # Decoder Path
        def decoder_block(x, skip, filters, dropout_rate=0.1):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.concatenate([x, skip])
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            return x

        # Encoder
        c1, p1 = encoder_block(inputs, 32)
        c2, p2 = encoder_block(p1, 64)
        c3, p3 = encoder_block(p2, 128)
        c4, p4 = encoder_block(p3, 256)

        # Bottleneck
        b = layers.Conv2D(512, (3, 3), padding='same')(p4)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('relu')(b)
        b = layers.Conv2D(512, (3, 3), padding='same')(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('relu')(b)

        # Decoder
        d1 = decoder_block(b, c4, 256)
        d2 = decoder_block(d1, c3, 128)
        d3 = decoder_block(d2, c2, 64)
        d4 = decoder_block(d3, c1, 32)

        # Output
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=dice_loss,
                      metrics=[dice_coef, 'accuracy', iou_metric])  # IoU metriği değiştirildi

        return model


# 3. Görüntü İşleme Sınıfı (Aynı)
class ImageProcessor:
    @staticmethod
    def load_image(image_path, target_size=(128, 128)):  # Boyutu 128x128 olarak değiştir
        """Görüntüyü yükler ve ön işleme yapar"""
        try:
            img = Image.open(image_path)

            # Görüntüyü gri tonlamaya çevir
            if img.mode != 'L':
                img = img.convert('L')

            # Histogram eşitleme
            img_array = np.array(img)
            img_array = exposure.equalize_hist(img_array)
            img = Image.fromarray((img_array * 255).astype(np.uint8))

            # Yeniden boyutlandır
            img = img.resize(target_size)

            # Normalizasyon ve boyut ayarı
            img_array = np.array(img) / 255.0
            return img_array.reshape(1, *target_size, 1)  # 1 kanal (grayscale) olarak döndür
        except Exception as e:
            raise ValueError(f"Görüntü yükleme hatası: {str(e)}")

    @staticmethod
    def post_process_mask(predicted_mask, original_size):
        """Model çıktısını işleyerek binary maske oluşturur"""
        # Model çıktısını orijinal boyuta dönüştür
        mask = cv2.resize(predicted_mask[0, :, :, 0], original_size)

        # Eşikleme
        mask_uint8 = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Gürültü temizleme
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

        # Küçük nesneleri kaldır
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                cv2.drawContours(processed_mask, [cnt], -1, 0, -1)

        return processed_mask


# 4. Ana GUI Uygulaması (Model yükleme kısmı değiştirildi)
class TumorSegmentationApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Gelişmiş Tümör Segmentasyon Sistemi")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        self.model_path = model_path
        self.model = None
        self.load_model()

        self.create_widgets()
        self.original_image = None
        self.processed_image = None
        self.mask = None
        self.image_path = None

    def load_model(self):
        """Modeli yükler veya yeni bir model oluşturur"""
        try:
            if os.path.exists(self.model_path):
                # Özel metrik fonksiyonlarını tanımla
                custom_objects = {
                    'dice_coef': dice_coef,
                    'dice_loss': dice_loss,
                    'iou_metric': iou_metric  # IoU metriği değiştirildi
                }

                # Modeli yükle
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects,
                    compile=False  # Önce compile=False ile yükle
                )

                # Modeli tekrar compile et
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=dice_loss,
                    metrics=[dice_coef, 'accuracy', iou_metric]  # IoU metriği değiştirildi
                )

                print("Model başarıyla yüklendi ve compile edildi.")
            else:
                messagebox.showwarning("Uyarı", "Model dosyası bulunamadı. Yeni bir model oluşturuluyor.")
                self.model = TumorSegmentationModel().model
                # Yeni modeli kaydet
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
                print(f"Yeni model oluşturuldu ve kaydedildi: {self.model_path}")
        except Exception as e:
            messagebox.showerror("Hata", f"Model yükleme hatası: {str(e)}")
            print(f"Hata detayı: {str(e)}")
            self.root.destroy()

    # Diğer metodlar aynı...
    def create_widgets(self):
        """GUI arayüzünü oluşturur"""
        # Ana çerçeve
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Kontrol paneli
        control_frame = ttk.LabelFrame(main_frame, text="Kontrol Paneli", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Dosya seç butonu
        self.btn_load = ttk.Button(control_frame, text="Görüntü Yükle", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        # Segmentasyon butonu
        self.btn_process = ttk.Button(control_frame, text="Segmentasyon Yap", state=tk.DISABLED,
                                      command=self.process_image)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        # Kaydet butonu
        self.btn_save = ttk.Button(control_frame, text="Sonucu Kaydet", state=tk.DISABLED, command=self.save_results)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Görüntü görüntüleme alanı
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        # Orijinal görüntü
        self.original_label = ttk.Label(image_frame, text="Orijinal Görüntü")
        self.original_label.pack()
        self.original_panel = ttk.Label(image_frame)
        self.original_panel.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Segmentasyon sonucu
        self.result_label = ttk.Label(image_frame, text="Segmentasyon Sonucu")
        self.result_label.pack()
        self.result_panel = ttk.Label(image_frame)
        self.result_panel.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Bilgi paneli
        self.info_text = tk.Text(main_frame, height=5, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X, pady=5)

        # İlerleme çubuğu
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

    def load_image(self):
        """Görüntü yüklemek için dosya iletişim kutusunu açar"""
        file_path = filedialog.askopenfilename(
            title="Tıbbi Görüntü Seçin",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.dcm"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path)

            # Görüntüyü göster
            self.display_image(self.original_image, self.original_panel)

            # Bilgi güncelle
            self.update_info(f"Yüklenen görüntü: {os.path.basename(file_path)}\n"
                             f"Boyut: {self.original_image.size}")

            # Buton durumlarını güncelle
            self.btn_process.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü yükleme hatası: {str(e)}")

    def process_image(self):
        """Görüntüyü işler ve segmentasyon yapar"""
        if not self.image_path or not self.original_image:
            return

        try:
            self.progress.start()

            # Orijinal boyutu sakla (sonradan post-processing için gerekli)
            original_size = self.original_image.size

            # Görüntüyü model için hazırla (128x128 grayscale)
            img_array = ImageProcessor.load_image(self.image_path)

            # Segmentasyon yap
            prediction = self.model.predict(img_array)

            # Post-processing (orijinal boyuta dönüştür)
            self.mask = ImageProcessor.post_process_mask(prediction, original_size)

            # Sonucu görselleştir
            self.visualize_results()

            # Bilgi güncelle
            tumor_area = np.sum(self.mask > 0) / (original_size[0] * original_size[1]) * 100
            self.update_info(f"Segmentasyon tamamlandı\n"
                             f"Tümör alanı: {tumor_area:.2f}%")

            # Buton durumlarını güncelle
            self.btn_save.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Hata", f"Segmentasyon hatası: {str(e)}")
        finally:
            self.progress.stop()

    def visualize_results(self):
        """Segmentasyon sonuçlarını görselleştirir"""
        original_img = np.array(self.original_image.convert('RGB'))
        mask_resized = cv2.resize(self.mask, (original_img.shape[1], original_img.shape[0]))

        # Maskeyi renklendir
        colored_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)

        # Orijinal görüntü ile maskeyi birleştir
        overlay = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)

        # Konturları çiz
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Sonuçları göster
        self.processed_image = Image.fromarray(overlay)
        self.display_image(self.processed_image, self.result_panel)

    def save_results(self):
        """Sonuçları kaydeder"""
        if not self.processed_image or not self.image_path:
            return

        save_dir = "segmentasyon_sonuclari"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]

        # Orijinal + segmentasyon birleşik görüntü
        combined = Image.new('RGB', (self.original_image.width * 2, self.original_image.height))
        combined.paste(self.original_image, (0, 0))
        combined.paste(self.processed_image, (self.original_image.width, 0))

        save_path = os.path.join(save_dir, f"{base_name}_{timestamp}.png")
        combined.save(save_path)

        # Maskeyi ayrı olarak kaydet
        mask_path = os.path.join(save_dir, f"{base_name}_{timestamp}_mask.png")
        Image.fromarray(self.mask).save(mask_path)

        messagebox.showinfo("Başarılı", f"Sonuçlar başarıyla kaydedildi:\n{save_path}")

    def display_image(self, image, panel):
        """Görüntüyü GUI'de gösterir"""
        # Görüntüyü panel boyutuna uygun şekilde yeniden boyutlandır
        width, height = self.get_display_size(image)
        img_resized = image.resize((width, height), Image.LANCZOS)

        # Tkinter için uygun formata dönüştür
        img_tk = ImageTk.PhotoImage(img_resized)

        # Paneli güncelle
        panel.config(image=img_tk)
        panel.image = img_tk  # Referansı koru

    def get_display_size(self, image):
        """Panel boyutuna göre görüntü boyutunu hesaplar"""
        panel_width = self.original_panel.winfo_width() or 400
        panel_height = self.original_panel.winfo_height() or 300

        img_ratio = image.width / image.height
        panel_ratio = panel_width / panel_height

        if img_ratio > panel_ratio:
            width = panel_width
            height = int(panel_width / img_ratio)
        else:
            height = panel_height
            width = int(panel_height * img_ratio)

        return max(width, 100), max(height, 100)  # Minimum boyut

    def update_info(self, text):
        """Bilgi panelini günceller"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)


# Uygulamayı başlat
# Uygulamayı başlat
if __name__ == "__main__":
    root = tk.Tk()

    # Model yolu - mutlak yol kullanmaya çalışın
    model_path = os.path.abspath("../models/unet_model.h5")
    print(f"Model yolu: {model_path}")

    # Model dosyasının varlığını kontrol et
    if not os.path.exists(model_path):
        print(f"Uyarı: Model dosyası bulunamadı: {model_path}")

    # Uygulamayı başlat
    app = TumorSegmentationApp(root, model_path=model_path)

    # Pencere boyutlandırma olaylarını dinle
    def on_resize(event):
        if hasattr(app, 'original_image') and app.original_image is not None:
            app.display_image(app.original_image, app.original_panel)
        if hasattr(app, 'processed_image') and app.processed_image is not None:
            app.display_image(app.processed_image, app.result_panel)

    root.bind("<Configure>", on_resize)

    root.mainloop()


    root.bind("<Configure>", on_resize)

    root.mainloop()