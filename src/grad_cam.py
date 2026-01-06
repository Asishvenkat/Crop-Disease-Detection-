"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for MobileNetV2
Generates visual explanations highlighting regions that influenced the model's prediction
"""
import numpy as np
import cv2
import tensorflow as tf
import keras
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for explaining CNN predictions
    Works with MobileNetV2 architecture
    """
    
    def __init__(self, model, layer_name="global_average_pooling2d"):
        """
        Initialize Grad-CAM
        
        Args:
            model: Keras model instance
            layer_name: Name of the convolutional layer to visualize
                       (default works with MobileNetV2's feature extraction layer)
        """
        self.model = model
        self.layer_name = layer_name
        self.conv_layer, self.grad_model = self._create_grad_model()
        self.grad_model_built = None
    
    def _create_grad_model(self):
        """Create a model that outputs gradients of predictions with respect to activations"""
        # Recursively find a target conv layer by name, otherwise fall back to the last Conv2D
        def find_target_conv(model_or_layer, target_name=None):
            if target_name and hasattr(model_or_layer, "name") and model_or_layer.name == target_name:
                return model_or_layer
            if isinstance(model_or_layer, keras.layers.Conv2D):
                candidate = model_or_layer
            else:
                candidate = None
            if hasattr(model_or_layer, "layers"):
                for layer in model_or_layer.layers:
                    result = find_target_conv(layer, target_name)
                    if result is not None:
                        candidate = result
            return candidate

        # Prefer the user-specified layer name if it exists; otherwise take the last conv layer
        conv_layer = find_target_conv(self.model, self.layer_name)
        if conv_layer is None:
            conv_layer = find_target_conv(self.model, None)
        if conv_layer is None:
            raise ValueError("No Conv2D layer found for Grad-CAM in model")

        print(f"[OK] Using layer for Grad-CAM: {conv_layer.name}")

        # Store both the conv layer and its name; the functional model is built lazily
        return conv_layer, conv_layer.name
    
    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap - actual gradient-based visualization
        
        Args:
            img_array: Input image array (160, 160, 3), values in [0, 1]
            pred_index: Index of the class to explain (if None, uses argmax)
        
        Returns:
            heatmap: Normalized heatmap (0-1) same size as input
        """
        # Ensure image is batch of size 1
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Build the grad model once after the main model has been called
        if self.grad_model_built is None:
            # Ensure the model is built by running a dummy forward pass
            try:
                _ = self.model(img_tensor, training=False)
            except Exception:
                pass

            target_layer = self.conv_layer
            # If the captured conv layer is missing, try to retrieve by name
            if target_layer is None:
                try:
                    target_layer = self.model.get_layer(self.grad_model)
                except Exception:
                    target_layer = None

            if target_layer is None:
                # Cannot proceed with Grad-CAM
                self.grad_model_built = None
            else:
                inputs = getattr(self.model, "inputs", None)
                if inputs is None or len(inputs) == 0:
                    # If still undefined, attempt another forward pass and recheck
                    try:
                        _ = self.model(img_tensor, training=False)
                        inputs = getattr(self.model, "inputs", None)
                    except Exception:
                        inputs = None

                if inputs is not None and len(inputs) > 0:
                    try:
                        self.grad_model_built = tf.keras.Model(
                            inputs=inputs,
                            outputs=[target_layer.output, self.model.output]
                        )
                    except Exception:
                        self.grad_model_built = None
                else:
                    self.grad_model_built = None
        
        heatmap_np = None
        saliency_np = None

        # Try Grad-CAM
        if self.grad_model_built is not None:
            try:
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = self.grad_model_built(img_tensor, training=False)
                    if pred_index is None:
                        pred_index = tf.argmax(predictions[0])
                    class_channel = predictions[:, pred_index]

                grads = tape.gradient(class_channel, conv_outputs)

                if grads is not None:
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_outputs = conv_outputs[0]
                    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
                    heatmap = tf.nn.relu(heatmap)

                    max_val = tf.reduce_max(heatmap)
                    if max_val > 0:
                        heatmap = heatmap / max_val

                    heatmap = tf.image.resize(
                        tf.expand_dims(tf.expand_dims(heatmap, 0), -1),
                        (img_array.shape[1], img_array.shape[2]),
                        method="bilinear"
                    )[0, :, :, 0]

                    heatmap_np = heatmap.numpy()
                    heatmap_np = cv2.GaussianBlur(heatmap_np, (3, 3), 0)
                    if heatmap_np.max() > 0:
                        heatmap_np = heatmap_np / heatmap_np.max()
            except Exception as e:
                print(f"[INFO] Grad-CAM failed: {e}")

        # Saliency on input (gradient w.r.t. input)
        try:
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                predictions = self.model(img_tensor, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

            grads = tape.gradient(class_channel, img_tensor)
            if grads is not None:
                saliency = tf.reduce_mean(tf.abs(grads[0]), axis=-1)
                saliency_np = saliency.numpy()
                saliency_np = cv2.GaussianBlur(saliency_np, (7, 7), 0)
                max_val = saliency_np.max()
                if max_val > 0:
                    saliency_np = saliency_np / max_val
        except Exception:
            saliency_np = saliency_np

        # If both available, combine for sharper focus (guided Grad-CAM style)
        if heatmap_np is not None and saliency_np is not None:
            combined = heatmap_np * saliency_np
            if combined.max() > 0:
                combined = combined / combined.max()
            return combined

        # Otherwise return whichever is available
        if heatmap_np is not None:
            return heatmap_np
        if saliency_np is not None:
            return saliency_np

        # Last-resort: uniform map (avoids misleading hotspots)
        return np.ones((img_array.shape[1], img_array.shape[2]), dtype=np.float32)
    
    def overlay_heatmap(self, original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image for visualization (OPTIMIZED)
        
        Args:
            original_img: Original RGB image, values in [0, 255]
            heatmap: Grad-CAM heatmap
            alpha: Transparency of heatmap overlay (default: 0.4)
            colormap: OpenCV colormap to use
        
        Returns:
            overlay: Overlaid image for visualization
        """
        from src.config import GRADCAM_RESOLUTION
        
        # Downsize for faster processing
        target_size = GRADCAM_RESOLUTION  # e.g., (128, 128)
        original_small = cv2.resize(original_img, target_size, interpolation=cv2.INTER_LINEAR)
        heatmap_small = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert heatmap to 0-255 range
        heatmap_8bit = np.uint8(255 * heatmap_small)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, colormap)
        
        # Convert original from RGB to BGR for OpenCV
        original_bgr = cv2.cvtColor(original_small, cv2.COLOR_RGB2BGR)
        
        # Blend images
        overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def explain_prediction(self, img_array, original_img, pred_index=None, alpha=0.5):
        """
        Complete explanation: generate heatmap and overlay on original image
        
        Args:
            img_array: Preprocessed image array (224, 224, 3), values in [0, 1]
            original_img: Original image for visualization
            pred_index: Class index to explain
            alpha: Heatmap overlay transparency (default: 0.4)
        
        Returns:
            dict: Contains heatmap and overlaid visualization
        """
        heatmap = self.generate_heatmap(img_array, pred_index)

        # Apply a simple leaf mask to suppress background speckle
        try:
            orig_uint8 = original_img
            if orig_uint8.dtype != np.uint8:
                if orig_uint8.max() <= 1.0:
                    orig_uint8 = (orig_uint8 * 255).astype(np.uint8)
                else:
                    orig_uint8 = orig_uint8.astype(np.uint8)

            gray = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2GRAY) if len(orig_uint8.shape) == 3 else orig_uint8
            # Otsu threshold to segment leaf
            _, leaf_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Smooth mask edges a bit
            leaf_mask = cv2.GaussianBlur(leaf_mask.astype(np.float32), (9, 9), 0)
            if leaf_mask.max() > 0:
                leaf_mask = leaf_mask / leaf_mask.max()

            # Resize mask to heatmap size if needed
            if leaf_mask.shape != heatmap.shape:
                leaf_mask = cv2.resize(leaf_mask, (heatmap.shape[1], heatmap.shape[0]))

            heatmap = heatmap * leaf_mask
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
        except Exception:
            pass

        overlay = self.overlay_heatmap(original_img, heatmap, alpha=alpha)
        
        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "explanation": "Red/hot colors indicate regions that strongly influenced the prediction"
        }


def save_explanation_image(overlay_img, output_path):
    """Save explanation image to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay_bgr)
    
    return output_path


print("[OK] Grad-CAM module loaded successfully")
