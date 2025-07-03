import os
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from skimage import exposure
from collections import deque


# 1. Custom metric functions
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def iou_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / (union + K.epsilon())


# 2. Advanced U-Net Model
class TumorSegmentationModel:
    def __init__(self, input_shape=(128, 128, 1)):
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
                      metrics=[dice_coef, 'accuracy', iou_metric])
        return model


# 3. Image Processor Class
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


# 4. Main GUI Application
class TumorSegmentationApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Advanced Tumor Segmentation System")
        self.root.geometry("1100x800")
        self.root.minsize(1000, 700)

        self.model_path = model_path
        self.model = None
        self.load_model()

        # Image variables
        self.original_image = None
        self.processed_image = None
        self.mask = None
        self.backup_mask = None
        self.image_path = None
        self.tk_image = None
        self.tk_processed_image = None

        # ROI selection variables
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_end_x = None
        self.rect_end_y = None
        self.rectangle_id = None
        self.selecting_roi = False

        # Zoom and pan variables
        self.zoom_level = 1.0
        self.zoom_factor = 1.2
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # Control points variables
        self.dragging_point = None
        self.points = []
        self.point_radius = 8
        self.point_ids = []
        self.contour_points = []
        self.edit_mode = False

        # Edit history variables
        self.edit_history = deque(maxlen=20)  # Stores past states
        self.edit_future = deque(maxlen=20)  # Stores undone states
        self.current_edit = None  # Current state

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        self.btn_load = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_select_roi = ttk.Button(control_frame, text="Select ROI", state=tk.DISABLED,
                                         command=self.enable_roi_selection)
        self.btn_select_roi.pack(side=tk.LEFT, padx=5)

        self.btn_process = ttk.Button(control_frame, text="Segment", state=tk.DISABLED,
                                      command=self.process_image)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        self.btn_edit = ttk.Button(control_frame, text="Edit", state=tk.DISABLED,
                                   command=self.start_editing)
        self.btn_edit.pack(side=tk.LEFT, padx=5)

        self.btn_save_edit = ttk.Button(control_frame, text="Save Edit", state=tk.DISABLED,
                                        command=self.save_edit)
        self.btn_save_edit.pack(side=tk.LEFT, padx=5)

        self.btn_cancel_edit = ttk.Button(control_frame, text="Cancel Edit", state=tk.DISABLED,
                                          command=self.cancel_edit)
        self.btn_cancel_edit.pack(side=tk.LEFT, padx=5)

        # Navigation buttons
        self.btn_undo = ttk.Button(control_frame, text="← Back", state=tk.DISABLED,
                                   command=self.undo_edit)
        self.btn_undo.pack(side=tk.LEFT, padx=5)

        self.btn_redo = ttk.Button(control_frame, text="Forward →", state=tk.DISABLED,
                                   command=self.redo_edit)
        self.btn_redo.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(control_frame, text="Save Results", state=tk.DISABLED,
                                   command=self.save_results)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Image display area
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        # Original image panel
        self.original_label = ttk.Label(image_frame, text="Original Image")
        self.original_label.pack()

        self.original_canvas = tk.Canvas(image_frame, bg='white', width=450, height=450)
        self.original_canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.original_canvas.bind("<ButtonPress-1>", self.start_roi_selection)
        self.original_canvas.bind("<B1-Motion>", self.update_roi_selection)
        self.original_canvas.bind("<ButtonRelease-1>", self.end_roi_selection)

        # Result panel with scrollbars
        result_frame = ttk.Frame(image_frame)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.result_label = ttk.Label(result_frame, text="Segmentation Result")
        self.result_label.pack()

        # Create canvas with scrollbars
        canvas_container = ttk.Frame(result_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        self.v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        self.h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Result canvas
        self.result_canvas = tk.Canvas(
            canvas_container,
            bg='white',
            width=450,
            height=450,
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set
        )
        self.result_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbars
        self.v_scroll.config(command=self.result_canvas.yview)
        self.h_scroll.config(command=self.result_canvas.xview)

        # Bind mouse wheel for zooming
        self.result_canvas.bind("<MouseWheel>", self.on_mousewheel)

        # Bind control point events
        self.result_canvas.bind("<ButtonPress-1>", self.on_point_press)
        self.result_canvas.bind("<B1-Motion>", self.on_point_drag)
        self.result_canvas.bind("<ButtonRelease-1>", self.on_point_release)

        # Info panel
        self.info_text = tk.Text(main_frame, height=5, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X, pady=5)

        # Zoom control
        zoom_frame = ttk.Frame(main_frame)
        zoom_frame.pack(fill=tk.X, pady=5)

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_slider = ttk.Scale(zoom_frame, from_=0.1, to=5.0, value=1.0,
                                     command=self.update_zoom_from_slider)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.zoom_label = ttk.Label(zoom_frame, text="1.0x")
        self.zoom_label.pack(side=tk.LEFT)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

    def save_current_state(self):
        """Save current state to history before making changes"""
        if self.mask is not None and self.points:
            state = {
                'mask': self.mask.copy(),
                'points': [p for p in self.points]
            }
            self.edit_history.append(state)
            self.current_edit = state
            self.edit_future.clear()  # Clear redo stack when new changes are made
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update the state of navigation buttons based on history"""
        self.btn_undo.config(state=tk.NORMAL if len(self.edit_history) > 0 else tk.DISABLED)
        self.btn_redo.config(state=tk.NORMAL if len(self.edit_future) > 0 else tk.DISABLED)

    def undo_edit(self):
        """Revert to previous state"""
        if len(self.edit_history) > 0:
            # Save current state to redo stack
            if self.current_edit:
                self.edit_future.append(self.current_edit)

            # Get previous state
            prev_state = self.edit_history.pop()
            self.mask = prev_state['mask'].copy()
            self.points = [p for p in prev_state['points']]
            self.current_edit = prev_state

            self.visualize_results()
            self.update_navigation_buttons()

    def redo_edit(self):
        """Redo an undone edit"""
        if len(self.edit_future) > 0:
            # Save current state to undo stack
            if self.current_edit:
                self.edit_history.append(self.current_edit)

            # Get next state
            next_state = self.edit_future.pop()
            self.mask = next_state['mask'].copy()
            self.points = [p for p in next_state['points']]
            self.current_edit = next_state

            self.visualize_results()
            self.update_navigation_buttons()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                custom_objects = {
                    'dice_coef': dice_coef,
                    'dice_loss': dice_loss,
                    'iou_metric': iou_metric
                }
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=dice_loss,
                    metrics=[dice_coef, 'accuracy', iou_metric]
                )
                print("Model loaded and compiled successfully.")
            else:
                messagebox.showwarning("Warning", "Model file not found. Creating new model.")
                self.model = TumorSegmentationModel().model
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
                print(f"New model created and saved: {self.model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Model loading error: {str(e)}")
            print(f"Error details: {str(e)}")
            self.root.destroy()

    def start_editing(self):
        if self.mask is None:
            messagebox.showwarning("Warning", "Please perform segmentation first")
            return

        self.edit_mode = True
        self.backup_mask = self.mask.copy()
        self.btn_edit.config(state=tk.DISABLED)
        self.btn_save_edit.config(state=tk.NORMAL)
        self.btn_cancel_edit.config(state=tk.NORMAL)

        # Clear any previous history
        self.edit_history.clear()
        self.edit_future.clear()
        self.current_edit = None

        # Save initial state
        self.save_current_state()

        self.update_info("Editing mode: Drag points to adjust the segmentation")
        self.visualize_results()

    def save_edit(self):
        self.edit_mode = False
        self.btn_edit.config(state=tk.NORMAL)
        self.btn_save_edit.config(state=tk.DISABLED)
        self.btn_cancel_edit.config(state=tk.DISABLED)
        self.btn_undo.config(state=tk.DISABLED)
        self.btn_redo.config(state=tk.DISABLED)
        self.backup_mask = None
        self.edit_history.clear()
        self.edit_future.clear()
        self.current_edit = None
        self.update_info("Edits saved successfully")

    def cancel_edit(self):
        if self.backup_mask is not None:
            self.edit_mode = False
            self.mask = self.backup_mask
            self.btn_edit.config(state=tk.NORMAL)
            self.btn_save_edit.config(state=tk.DISABLED)
            self.btn_cancel_edit.config(state=tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)
            self.btn_redo.config(state=tk.DISABLED)
            self.edit_history.clear()
            self.edit_future.clear()
            self.current_edit = None
            self.update_info("Edits canceled")
            self.visualize_results()

    def on_point_press(self, event):
        if not self.edit_mode or not self.points:
            return

        x = self.result_canvas.canvasx(event.x)
        y = self.result_canvas.canvasy(event.y)

        for i, point in enumerate(self.points):
            px = point[0] * self.zoom_level
            py = point[1] * self.zoom_level
            distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5

            if distance <= self.point_radius:
                self.dragging_point = i
                # Save state before making changes
                self.save_current_state()
                break

    def on_point_drag(self, event):
        if not self.edit_mode or self.dragging_point is None:
            return

        new_x = self.result_canvas.canvasx(event.x) / self.zoom_level
        new_y = self.result_canvas.canvasy(event.y) / self.zoom_level

        # Update only the dragged point
        self.points[self.dragging_point] = (new_x, new_y)

        # Update the mask based on new points
        self.update_mask_from_points()

    def update_mask_from_points(self):
        """Update the mask based on current control points"""
        if not self.points:
            return

        temp_mask = np.zeros_like(self.mask)
        contour = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

        cv2.drawContours(temp_mask, [contour], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(contour)
        expanded_rect = (
            max(0, x - 10), max(0, y - 10),
            min(self.mask.shape[1], x + w + 10),
            min(self.mask.shape[0], y + h + 10)
        )

        self.mask[expanded_rect[1]:expanded_rect[3], expanded_rect[0]:expanded_rect[2]] = \
            temp_mask[expanded_rect[1]:expanded_rect[3], expanded_rect[0]:expanded_rect[2]]

        self.visualize_results()

    def on_point_release(self, event):
        if self.dragging_point is not None:
            # After dragging is complete, save the final state
            self.save_current_state()
        self.dragging_point = None

    def enable_roi_selection(self):
        if not self.original_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        self.selecting_roi = True
        self.btn_select_roi.config(state=tk.DISABLED)
        self.btn_process.config(state=tk.DISABLED)
        messagebox.showinfo("Info", "Please draw a rectangle on the image to select ROI")

    def start_roi_selection(self, event):
        if self.selecting_roi:
            self.rect_start_x = event.x
            self.rect_start_y = event.y
            self.rect_end_x = event.x
            self.rect_end_y = event.y
            self.rectangle_id = self.original_canvas.create_rectangle(
                self.rect_start_x, self.rect_start_y,
                self.rect_end_x, self.rect_end_y,
                outline='red', width=2
            )

    def update_roi_selection(self, event):
        if self.selecting_roi and self.rectangle_id:
            self.rect_end_x = event.x
            self.rect_end_y = event.y
            self.original_canvas.coords(
                self.rectangle_id,
                self.rect_start_x, self.rect_start_y,
                self.rect_end_x, self.rect_end_y
            )

    def end_roi_selection(self, event):
        if self.selecting_roi:
            self.selecting_roi = False
            self.btn_process.config(state=tk.NORMAL)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.dcm"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path)

            # Clear previous ROI selection
            self.rect_start_x = None
            self.rect_start_y = None
            self.rect_end_x = None
            self.rect_end_y = None
            self.original_canvas.delete("all")

            # Reset zoom and pan
            self.zoom_level = 1.0
            self.zoom_slider.set(1.0)
            self.zoom_label.config(text="1.0x")
            self.pan_offset_x = 0
            self.pan_offset_y = 0

            # Display image
            self.tk_image = ImageTk.PhotoImage(self.original_image)
            self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            # Update info
            self.update_info(f"Loaded image: {os.path.basename(file_path)}\n"
                             f"Size: {self.original_image.size}")

            # Update button states
            self.btn_select_roi.config(state=tk.NORMAL)
            self.btn_process.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED)
            self.btn_edit.config(state=tk.DISABLED)
            self.btn_save_edit.config(state=tk.DISABLED)
            self.btn_cancel_edit.config(state=tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)
            self.btn_redo.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Image loading error: {str(e)}")

    def process_image(self):
        if not self.image_path or not self.original_image:
            return

        if not all([self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y]):
            messagebox.showwarning("Warning", "Please select an ROI first")
            return

        try:
            self.progress.start()

            # Get image dimensions
            img_width, img_height = self.original_image.size
            original_img = np.array(self.original_image.convert('RGB'))

            # Get ROI coordinates (ensure they're within image bounds)
            x1 = max(0, min(self.rect_start_x, self.rect_end_x, img_width - 1))
            y1 = max(0, min(self.rect_start_y, self.rect_end_y, img_height - 1))
            x2 = min(img_width - 1, max(self.rect_start_x, self.rect_end_x, 0))
            y2 = min(img_height - 1, max(self.rect_start_y, self.rect_end_y, 0))

            # Extract ROI
            roi = original_img[y1:y2, x1:x2]
            if roi.size == 0:
                raise ValueError("Selected ROI is too small or invalid")

            # Save ROI to temp file and process
            temp_path = "temp_roi.png"
            Image.fromarray(roi).save(temp_path)
            img_array = ImageProcessor.load_image(temp_path)

            # Predict
            prediction = self.model.predict(img_array)

            # Process mask
            roi_height, roi_width = roi.shape[:2]
            mask = ImageProcessor.post_process_mask(prediction, (roi_width, roi_height))

            # Create full-size mask
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask
            self.mask = full_mask

            # Visualize results
            self.visualize_results()

            # Update info
            tumor_area = np.sum(self.mask > 0) / (img_width * img_height) * 100
            self.update_info(f"Segmentation complete\n"
                             f"Tumor area: {tumor_area:.2f}%\n"
                             f"ROI Coordinates: ({x1},{y1})-({x2},{y2})")

            # Update buttons
            self.btn_save.config(state=tk.NORMAL)
            self.btn_select_roi.config(state=tk.NORMAL)
            self.btn_edit.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Segmentation error: {str(e)}")
        finally:
            self.progress.stop()
            if os.path.exists("temp_roi.png"):
                os.remove("temp_roi.png")

    def visualize_results(self):
        if self.original_image is None or self.mask is None:
            return

        original_img = np.array(self.original_image.convert('RGB'))
        colored_mask = cv2.applyColorMap(self.mask, cv2.COLORMAP_JET)

        # Create overlay
        overlay = original_img.copy()
        overlay[self.mask > 0] = overlay[self.mask > 0] * 0.7 + colored_mask[self.mask > 0] * 0.3

        # Draw ROI rectangle
        if all([self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y]):
            x1 = min(self.rect_start_x, self.rect_end_x)
            y1 = min(self.rect_start_y, self.rect_end_y)
            x2 = max(self.rect_start_x, self.rect_end_x)
            y2 = max(self.rect_start_y, self.rect_end_y)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Find contours and create control points
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Update control points with simplified contour
        self.points = []
        for cnt in contours:
            # Simplify contour to reduce number of points
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            for point in approx:
                self.points.append((point[0][0], point[0][1]))

        # Display result
        self.processed_image = Image.fromarray(overlay)
        self.display_result_image()

    def display_result_image(self):
        self.result_canvas.delete("all")
        if not self.processed_image:
            return

        img_width, img_height = self.processed_image.size
        zoomed_width = int(img_width * self.zoom_level)
        zoomed_height = int(img_height * self.zoom_level)

        # Resize image
        resized_img = self.processed_image.resize((zoomed_width, zoomed_height), Image.LANCZOS)
        self.tk_processed_image = ImageTk.PhotoImage(resized_img)

        # Update scroll region
        self.result_canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))

        # Add image to canvas
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_processed_image)

        # Draw control points if in edit mode
        if self.edit_mode:
            self.draw_control_points()

    def draw_control_points(self):
        self.point_ids = []
        for point in self.points:
            x = point[0] * self.zoom_level
            y = point[1] * self.zoom_level

            point_id = self.result_canvas.create_oval(
                x - self.point_radius, y - self.point_radius,
                x + self.point_radius, y + self.point_radius,
                fill="red", outline="white", width=2
            )
            self.point_ids.append(point_id)

    def update_zoom_from_slider(self, value):
        try:
            self.zoom_level = float(value)
            self.zoom_label.config(text=f"{self.zoom_level:.1f}x")
            self.display_result_image()
        except ValueError:
            pass

    def on_mousewheel(self, event):
        # Determine zoom direction
        if event.delta > 0:
            self.zoom_level *= self.zoom_factor
        else:
            self.zoom_level /= self.zoom_factor

        # Limit zoom levels
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))

        # Update slider and label
        self.zoom_slider.set(self.zoom_level)
        self.zoom_label.config(text=f"{self.zoom_level:.1f}x")

        self.display_result_image()

    def save_results(self):
        if not self.processed_image or not self.image_path:
            return

        save_dir = "segmentation_results"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]

        # Save combined image
        combined = Image.new('RGB', (self.original_image.width * 2, self.original_image.height))
        combined.paste(self.original_image, (0, 0))
        combined.paste(self.processed_image, (self.original_image.width, 0))

        save_path = os.path.join(save_dir, f"{base_name}_{timestamp}.png")
        combined.save(save_path)

        # Save mask separately
        mask_path = os.path.join(save_dir, f"{base_name}_{timestamp}_mask.png")
        Image.fromarray(self.mask).save(mask_path)

        messagebox.showinfo("Success", f"Results saved successfully:\n{save_path}")

    def update_info(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    model_path = os.path.abspath("unet_model.h5")

    if not os.path.exists(model_path):
        new_model = TumorSegmentationModel().model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        new_model.save(model_path)

    app = TumorSegmentationApp(root, model_path=model_path)
    root.mainloop()