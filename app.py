# app.py
from flask import Flask, request, render_template
import numpy as np
import cv2
import base64
import json
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Create a reverse dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            max_dimension = 1000
            if img.shape[0] > max_dimension or img.shape[1] > max_dimension:
                scaling_factor = max_dimension / float(max(img.shape[0], img.shape[1]))
                img = cv2.resize(img, (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))

            img_resized = cv2.resize(img, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(img_array)
            result_idx = np.argmax(prediction, axis=1)[0]
            result = class_names.get(result_idx, "Unknown")

            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            img_data_url = f"data:image/jpeg;base64,{img_base64}"
            result = result.replace('_', ' ')

            return render_template('result.html', result=result, image_data=img_data_url)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
