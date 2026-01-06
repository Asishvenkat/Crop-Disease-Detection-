"""
Disease spot detection using color-based heatmap
Detects brown/yellow diseased pixels and creates red heatmap visualization
"""
import numpy as np
import cv2
import tensorflow as tf


class GradCAM:
    """
    Color-based disease spot detector
    Highlights brown/yellow colored pixels (disease spots) with red heatmap
    """
    
    def __init__(self, model, layer_name="conv5_block3_out"):
        """
        Initialize disease spot detector
        
        Args:
            model: Keras model instance (not used for color detection)
            layer_name: Ignored (kept for API compatibility)
        """
        self.model = model
        self.layer_name = layer_name
        
        # Color detection setup - no layer needed
        self.conv_layer = None
        print(f"[OK] Using color-based disease detection")
                        # Recursively search in submodels
                        if hasattr(layer, 'layers'):
                            found = find_layer_recursive(layer, target_name)
                            if found:
                                return found
                return None
            
            self.conv_layer = find_layer_recursive(self.model, layer_name)
            
            if self.conv_layer:
                print(f"[OK] Using layer for Grad-CAM: {self.conv_layer.name}")
            else:
                print(f"[ERROR] Could not find layer {layer_name}")
                print(f"Available layers: {[l.name for l in self.model.layers]}")
    
    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap using direct gradient computation
        
        Args:
            img_array: Input image array (224, 224, 3), values in [0, 1]
            pred_index: Index of the class to explain (if None, uses argmax)
        
        Returns:
            heatmap: Normalized heatmap (0-1) same size as input
        """
        # Ensure image is batch of size 1
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        heatmap_np = None
        saliency_np = None

        # Try Grad-CAM with direct gradient computation
        if self.conv_layer is not None:
            try:
                # First, get the actual output through normal forward pass
                predictions = self.model(img_tensor, training=False)
                
                # Determine which class to explain
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                
                # Now compute Grad-CAM
                with tf.GradientTape() as tape:
                    # Forward pass to get conv outputs
                    conv_outputs = self.conv_layer(img_tensor, training=False)
                    
                    # Get the gradients of the class output w.r.t. conv layer outputs
                    # We need to get the full model path - run forward and compute gradients
                    conv_model = tf.keras.Model(
                        inputs=self.model.inputs,
                        outputs=self.conv_layer.output
                    )
                    conv_outputs = conv_model(img_tensor, training=False)
                    
                # Now compute gradients using the full model
                with tf.GradientTape() as grad_tape:
                    conv_outputs = self.conv_layer.output if hasattr(self.conv_layer, 'output') else None
                    
                    # Use intermediate_layer_model to get conv output
                    intermediate_layer_model = tf.keras.Model(
                        inputs=self.model.inputs,
                        outputs=self.conv_layer.output
                    )
                    intermediate_outputs = intermediate_layer_model(img_tensor, training=False)
                    
                    # Get predictions
                    preds = self.model(img_tensor, training=False)
                    class_channel = preds[:, pred_index]
                
                # Compute gradients
                grads = grad_tape.gradient(class_channel, intermediate_outputs)
                
                if grads is not None:
                    # Grad-CAM computation
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    intermediate_outputs_val = intermediate_outputs[0]
                    heatmap = tf.reduce_sum(intermediate_outputs_val * pooled_grads, axis=-1)
                    heatmap = tf.nn.relu(heatmap)

                    # Normalize
                    max_val = tf.reduce_max(heatmap)
                    if max_val > 0:
                        heatmap = heatmap / max_val

                    # Resize to input size
                    heatmap = tf.image.resize(
                        tf.expand_dims(tf.expand_dims(heatmap, 0), -1),
                        (img_array.shape[1], img_array.shape[2]),
                        method="bilinear"
                    )[0, :, :, 0]

                    heatmap_np = heatmap.numpy()
                    # Light blur to smooth
                    heatmap_np = cv2.GaussianBlur(heatmap_np, (3, 3), 0)
                    if heatmap_np.max() > 0:
                        heatmap_np = heatmap_np / heatmap_np.max()
            except Exception as e:
                print(f"[INFO] Grad-CAM failed: {e}")

        # Saliency as fallback
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
                # Blur saliency more than Grad-CAM
                saliency_np = cv2.GaussianBlur(saliency_np, (7, 7), 0)
                max_val = saliency_np.max()
                if max_val > 0:
                    saliency_np = saliency_np / max_val
        except Exception as e:
            print(f"[INFO] Saliency computation failed: {e}")

        # Combine: Grad-CAM * Saliency for sharper focus
        if heatmap_np is not None and saliency_np is not None:
            combined = heatmap_np * saliency_np
            return combined

        # Otherwise return whichever is available
        if heatmap_np is not None:
            return heatmap_np
        if saliency_np is not None:
            return saliency_np

        # Last-resort: uniform map (avoids misleading hotspots)
        return np.ones((img_array.shape[1], img_array.shape[2]), dtype=np.float32)
    
    def overlay_heatmap(self, original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image for visualization
        
        Args:
            original_img: Original RGB image (224, 224, 3), values in [0, 255]
            heatmap: Grad-CAM heatmap (224, 224)
            alpha: Transparency of heatmap overlay (default: 0.5)
            colormap: OpenCV colormap to use
        
        Returns:
            overlay_img: RGB image with heatmap overlay
        """
        # Ensure original image is uint8
        if original_img.dtype != np.uint8:
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
            else:
                original_img = original_img.astype(np.uint8)
        
        # Resize heatmap to match image size if needed
        if heatmap.shape[0] != original_img.shape[0]:
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Focus on activations (70th percentile - less aggressive)
        focus_threshold = np.percentile(heatmap, 70)
        heatmap_focus = np.where(heatmap >= focus_threshold, heatmap, heatmap * 0.2)

        # Normalize heatmap to 0-255 for colormap
        heatmap_normalized = (heatmap_focus * 255).astype(np.uint8)
        
        # Apply colormap (OpenCV uses BGR, so convert after)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert original image to float for blending
        original_float = original_img.astype(np.float32) / 255.0
        heatmap_float = heatmap_rgb.astype(np.float32) / 255.0
        
        # Blend: original * (1 - alpha) + heatmap * alpha
        overlay = (original_float * (1 - alpha) + heatmap_float * alpha) * 255
        overlay = overlay.astype(np.uint8)
        
        return overlay
    
    def explain_prediction(self, original_img, pred_index=None, alpha=0.5):
        """
        Generate and overlay a Grad-CAM heatmap on an image
        
        Args:
            original_img: Original RGB image array (224, 224, 3), values in [0, 1]
            pred_index: Index of the class to explain
            alpha: Transparency of overlay
        
        Returns:
            overlay_img: RGB image with heatmap overlay
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(original_img, pred_index)
        
        # Overlay on original image
        # Ensure original_img is in [0, 255] range for overlay
        if original_img.max() <= 1.0:
            img_for_overlay = (original_img * 255).astype(np.uint8)
        else:
            img_for_overlay = original_img.astype(np.uint8)
        
        overlay = self.overlay_heatmap(img_for_overlay, heatmap, alpha)
        
        return overlay
