import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image


def pil_read_grayscale(image_path):
    try:
        img = Image.open(image_path).convert('L')  # Gri ton
        img_np = np.array(img)
        return img_np
    except Exception as e:
        print(f"PIL ile okuma hatası: {e}")
        return None


def load_image(image_path, size=(128, 128)):
    img = pil_read_grayscale(image_path)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    img = cv2.resize(img, size) / 255.0
    return img.reshape(1, size[0], size[1], 1)


def process_and_save_mask(predicted_mask, original_image, folder="masks"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    mask = predicted_mask[0, :, :, 0]
    mask_uint8 = (mask * 255).astype(np.uint8)

    thresh_val = max(np.mean(mask_uint8) + 20, 150)
    _, binary_mask = cv2.threshold(mask_uint8, thresh_val, 255, cv2.THRESH_BINARY)

    binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(original_image.shape) == 2:
        original_image_colored = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        original_image_colored = original_image.copy()

    cv2.drawContours(original_image_colored, contours, -1, (0, 255, 0), 3)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(folder, f"contour_{timestamp}.png")
    cv2.imwrite(save_path, original_image_colored)
    print(f"Sonuç kaydedildi: {save_path}")


class TumorSegmentationApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Tümör Segmentasyon")
        self.root.geometry("450x180")

        self.model_path = model_path
        if not os.path.exists(self.model_path):
            messagebox.showerror("Hata", f"Model dosyası bulunamadı: {self.model_path}")
            self.root.destroy()
            return

        self.model = load_model(self.model_path)

        self.label = tk.Label(root, text="Tümörlü Resmi Yükle", font=("Arial", 14))
        self.label.pack(pady=10)

        self.file_path_var = tk.StringVar()
        self.file_path_label = tk.Label(root, textvariable=self.file_path_var, fg="blue", wraplength=400)
        self.file_path_label.pack(pady=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.load_button = tk.Button(btn_frame, text="Dosya Seç", width=15, command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.confirm_button = tk.Button(btn_frame, text="Onayla", width=15, command=self.run_prediction, state=tk.DISABLED)
        self.confirm_button.pack(side=tk.LEFT, padx=5)

        self.selected_image_path = None
        self.selected_image = None

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Tümörlü resmi seçin",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("Tüm dosyalar", "*.*")]
        )
        if not file_path:
            return

        img = pil_read_grayscale(file_path)
        if img is None:
            messagebox.showerror("Hata", "Seçilen dosya bir resim değil veya okunamadı (PIL).")
            self.file_path_var.set("")
            self.confirm_button.config(state=tk.DISABLED)
            return

        self.selected_image_path = file_path
        self.selected_image = img
        self.file_path_var.set(file_path)
        self.confirm_button.config(state=tk.NORMAL)

    def run_prediction(self):
        if self.selected_image_path is None or self.selected_image is None:
            messagebox.showwarning("Uyarı", "Önce dosya seçmelisiniz.")
            return

        try:
            image = load_image(self.selected_image_path, size=(128, 128))
            prediction = self.model.predict(image)
            process_and_save_mask(prediction, self.selected_image)
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu:\n{e}")


def main():
    root = tk.Tk()
    app = TumorSegmentationApp(root, model_path='../models/unet_model.h5')
    root.mainloop()


if __name__ == "__main__":
    main()
