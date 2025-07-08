import cv2
import numpy as np
from PIL import Image
from skimage import exposure

class ImageProcessor:
    @staticmethod
    def load_image(image_path, target_size=(128, 128)):
        try:
            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')

            img_array = np.array(img)
            img_array = exposure.equalize_hist(img_array)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img = img.resize(target_size)

            img_array = np.array(img) / 255.0
            return img_array.reshape(1, *target_size, 1)
        except Exception as e:
            raise ValueError(f"Image loading error: {str(e)}")

    @staticmethod
    def post_process_mask(predicted_mask, original_size):
        mask = cv2.resize(predicted_mask[0, :, :, 0], original_size)
        mask_uint8 = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                cv2.drawContours(processed_mask, [cnt], -1, 0, -1)
        return processed_mask