from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define the pix2pix upsample function
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

# Define the MobileNetV2-based U-Net model
def unet_model(input_size=(256, 256, 3), num_classes=59):
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_size,
        classes=num_classes,
        name="base_encoder_model"
    )
    base_model_layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in base_model_layer_names]
    encoder_model = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    encoder_model.trainable = False

    inputs = layers.Input(shape=input_size)
    enc_layers = encoder_model(inputs)

    x = upsample(512, 3, apply_dropout=True)(enc_layers[-1])
    x = layers.Concatenate()([x, enc_layers[-2]])
    x = upsample(256, 3, apply_dropout=True)(x)
    x = layers.Concatenate()([x, enc_layers[-3]])
    x = upsample(128, 3)(x)
    x = layers.Concatenate()([x, enc_layers[-4]])
    x = upsample(64, 3)(x)
    x = layers.Concatenate()([x, enc_layers[-5]])
    outputs = layers.Conv2DTranspose(num_classes, (3, 3), strides=2, padding="same", activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

# Load the trained model
model = unet_model(input_size=(256, 256, 3), num_classes=59)
model.load_weights('static/models/model_cloths.h5')

# Load class labels
class_names = pd.read_csv('static/labels.csv')
class_dict = dict(class_names.values)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'})
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clothing.jpg')
    file.save(image_path)
    
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({'success': False, 'message': 'Failed to load image'})
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    
    mask_pred = model.predict(img_input)[0]  # [256, 256, 59]
    mask = np.argmax(mask_pred, axis=-1).astype(np.uint8)  # [256, 256]
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Save full mask for debugging
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_mask.png'), mask * 4)
    
    # Detect all unique classes (excluding background, class 0)
    unique_classes = np.unique(mask)
    clothing_items = []
    
    for class_idx in unique_classes:
        if class_idx == 0:  # Skip background
            continue
        if class_idx not in class_dict:
            continue  # Skip if class not in labels.csv
        
        # Create mask for this class
        item_mask = (mask == class_idx).astype(np.uint8) * 255
        alpha = item_mask
        b, g, r = cv2.split(img)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        
        # Save the segmented item
        item_filename = f'segmented_item_{class_idx}.png'
        item_path = os.path.join(app.config['UPLOAD_FOLDER'], item_filename)
        cv2.imwrite(item_path, dst)
        
        # Add to response
        clothing_items.append({
            'label': class_dict[class_idx],
            'url': f'/static/uploads/{item_filename}'
        })
    
    if not clothing_items:
        return jsonify({'success': False, 'message': 'No clothing items detected'})
    
    return jsonify({'success': True, 'clothing_items': clothing_items})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)