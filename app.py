# app.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from preprocess import extract_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load model and set up label encoder ---
# You can choose between best_model.h5 or ser_cnn_model.h5
MODEL_PATH = "best_model.h5"  # or "ser_cnn_model.h5" if you prefer

try:
    model = load_model(MODEL_PATH)
    
    # Set up the label encoder (same order as training)
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    emotion_labels = list(emotion_map.values())
    label_encoder = LabelEncoder()
    label_encoder.fit(emotion_labels)
    
    # Debug: Print label encoder mapping
    print("Label encoder mapping:")
    for i, emotion in enumerate(label_encoder.classes_):
        print(f"{i}: {emotion}")
    
    print(f"Model loaded successfully from {MODEL_PATH}!")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract features using your existing preprocess.py
            mfcc = extract_features(filepath, max_pad_len=174)
            
            if mfcc is not None:
                X_input = mfcc.reshape(1, 40, 174, 1)  # add batch and channel dims
                
                # Predict
                prediction = model.predict(X_input)
                
                # Debug: Print raw prediction values
                print(f"Raw prediction values: {prediction[0]}")
                
                # Get confidence scores for each emotion
                confidence_scores = prediction[0].tolist()
                emotion_scores = {}
                for i, emotion in enumerate(emotion_labels):
                    emotion_scores[emotion] = float(confidence_scores[i])
                
                # Debug: Print scores
                print(f"Emotion scores: {emotion_scores}")
                
                # Find the emotion with highest confidence directly from the scores
                highest_emotion = max(emotion_scores, key=emotion_scores.get)
                highest_score = emotion_scores[highest_emotion]
                
                print(f"Highest emotion: {highest_emotion} with score {highest_score}")
                
                # Compare with argmax approach
                predicted_class = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                print(f"Argmax index: {predicted_class}, label: {predicted_label}")
                
                # Use the directly identified highest emotion
                return jsonify({
                    'prediction': highest_emotion,
                    'confidence': float(highest_score),
                    'all_scores': emotion_scores
                })
            else:
                return jsonify({'error': 'Failed to process the audio file'}), 400
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file format. Please upload a WAV file'}), 400

if __name__ == '__main__':
    app.run(debug=True)