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
        Generate Integrated Gradients saliency with leaf masking
        More accurate than raw gradients - focuses on disease regions
        
        Args:
            img_array: Input image array (H, W, 3), values in [0, 1]
            pred_index: Index of the class to explain (if None, uses argmax)
        
        Returns:
            heatmap: Normalized heatmap (0-1) same size as input
        """
        # Ensure image is batch of size 1
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        try:
            # Create baseline (black image)
            baseline = tf.zeros_like(img_tensor)
            
            # Integrated Gradients: interpolate between baseline and input
            num_steps = 20
            alphas = tf.linspace(0.0, 1.0, num_steps + 1)
            
            # Get prediction index
            predictions = self.model(img_tensor, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            integrated_grads = None
            
            for alpha in alphas:
                interpolated = baseline + alpha * (img_tensor - baseline)
                
                with tf.GradientTape() as tape:
                    tape.watch(interpolated)
                    preds = self.model(interpolated, training=False)
                    target_class = preds[:, pred_index]
                
                grads = tape.gradient(target_class, interpolated)
                
                if grads is not None:
                    if integrated_grads is None:
                        integrated_grads = grads
                    else:
                        integrated_grads += grads
            
            if integrated_grads is None:
                print("[WARNING] Integrated gradients failed")
                return np.ones((img_array.shape[1], img_array.shape[2]), dtype=np.float32)
            
            # Average and scale by input difference
            integrated_grads = integrated_grads / (num_steps + 1)
            integrated_grads = (img_tensor - baseline) * integrated_grads
            
            # Sum across RGB channels (positive contributions only)
            attribution = tf.reduce_sum(tf.abs(integrated_grads[0]), axis=-1)
            saliency_np = attribution.numpy()
            
            # Create leaf mask to suppress background
            img_uint8 = (img_array[0] * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Threshold to get leaf mask (green areas)
            _, leaf_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean mask
            kernel = np.ones((5, 5), np.uint8)
            leaf_mask = cv2.morphologyEx(leaf_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to saliency
            saliency_np = saliency_np * leaf_mask.astype(np.float32)
            
            # Smooth with moderate blur
            saliency_np = cv2.GaussianBlur(saliency_np, (11, 11), 1.5)
            
            # Enhance contrast
            if saliency_np.max() > 0:
                saliency_np = saliency_np / saliency_np.max()
                # Apply gamma correction to brighten disease spots
                saliency_np = np.power(saliency_np, 0.7)
            
            return saliency_np.astype(np.float32)
            
        except Exception as e:
            print(f"[ERROR] Integrated gradients failed: {e}")
            import traceback
            traceback.print_exc()
            return np.ones((img_array.shape[1], img_array.shape[2]), dtype=np.float32)
    
    def overlay_heatmap(self, original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_TURBO):
        """
        Overlay heatmap on original image with better visibility
        
        Args:
            original_img: Original RGB image, values in [0, 255]
            heatmap: Saliency heatmap (0-1)
            alpha: Heatmap transparency (0.5 for better blend)
            colormap: OpenCV colormap (TURBO for better contrast)
        
        Returns:
            overlay: Blended visualization (BGR, uint8)
        """
        from src.config import GRADCAM_RESOLUTION
        
        target_size = GRADCAM_RESOLUTION  # e.g., (128, 128)
        original_small = cv2.resize(original_img, target_size, interpolation=cv2.INTER_LINEAR)
        heatmap_small = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Ensure heatmap is in [0, 1]
        if heatmap_small.max() > 1.0:
            heatmap_small = heatmap_small / heatmap_small.max()
        
        # Convert heatmap to 0-255
        heatmap_8bit = np.uint8(255 * heatmap_small)
        
        # Apply colormap (TURBO has better visual contrast than JET)
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, colormap)
        
        # Convert original from RGB to BGR for OpenCV
        original_bgr = cv2.cvtColor(original_small, cv2.COLOR_RGB2BGR)
        
        # Blend images with better alpha for visibility
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
