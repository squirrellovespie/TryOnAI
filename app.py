import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras import layers, models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'static/models'

# Ensure upload and model directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Define the U-Net model architecture (must match the trained model)
def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    return models.Model(inputs, outputs)

# Load the pre-trained U-Net model
model = unet_model()
model.load_weights('models/unet_clothing_segmentation.h5')

# Function to segment clothing from an image
def segment_clothing(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    mask_pred = model.predict(img_input)
    mask_binary = (mask_pred > 0.5).astype(np.uint8)
    img_original_resized = cv2.resize(img, (128, 128))
    segmented_clothing = cv2.bitwise_and(img_original_resized, img_original_resized, mask=mask_binary[0, :, :, 0])
    segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented_' + os.path.basename(image_path))
    cv2.imwrite(segmented_path, segmented_clothing)
    return segmented_path

# Placeholder function to generate a 3D model
def generate_3d_model(segmented_images):
    # This is a placeholder; replace with actual 3D reconstruction logic (e.g., PIFuHD)
    # For now, it creates a dummy .obj file
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'clothing_model.obj')
    with open(model_path, 'w') as f:
        f.write("# Dummy OBJ file\nv 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3")
    return model_path

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('images')
        segmented_images = []
        for file in files:
            if file and file.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                segmented_path = segment_clothing(file_path)
                segmented_images.append(segmented_path)
        # Generate 3D model from segmented images
        model_path = generate_3d_model(segmented_images)
        return redirect(url_for('result', model_path=os.path.basename(model_path)))
    return render_template('index.html')

@app.route('/result')
def result():
    model_path = request.args.get('model_path')
    return render_template('result.html', model_path=model_path)

if __name__ == '__main__':
    app.run(debug=True)