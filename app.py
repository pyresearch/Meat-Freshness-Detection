from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
import traceback
import supervision as sv
import pyresearch

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the pre-trained YOLO model
try:
    model = YOLO('best.pt')  # Path to your best.pt model
    print("Model loaded successfully:", model.names)
except Exception as e:
    print("Error loading model:", str(e))
    model = None

# Directory to save uploaded files
UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1 limit

# Initialize supervision annotators with custom settings
box_annotator = sv.BoxAnnotator(thickness=1)  # Thinner bounding box
label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2, text_position=sv.Position.TOP_CENTER)  # Larger, centered labels

# Function to shrink bounding box
def shrink_box(box, scale=0.8):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    new_width = width * scale
    new_height = height * scale
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_x1 = center_x - new_width / 2
    new_x2 = center_x + new_width / 2
    new_y1 = center_y - new_height / 2
    new_y2 = center_y + new_height / 2
    return [new_x1, new_y1, new_x2, new_y2]

# Function to process a single image
def process_image(image_path):
    try:
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image:", image_path)
            return {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}, 'No objects detected. Please try another image.', None, None
        
        # Preprocess image for model
        model_input = cv2.resize(image, (640, 640))  # Adjust size based on model requirements
        results = model(model_input)  # Run inference
        print("Model inference results:", results)

        class_counts = {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}
        detected_class = None
        
        # Handle detection model output
        detections = sv.Detections.from_ultralytics(results[0])
        if len(detections) > 0:
            class_id = int(detections.class_id[0])
            if class_id < len(model.names):
                class_name = model.names[class_id]
                class_counts[class_name] = 1
                detected_class = class_name
                print(f"Detection result: {class_name}")
                # Shrink bounding box
                detections.xyxy[0] = shrink_box(detections.xyxy[0], scale=0.8)
            else:
                print("Invalid class ID:", class_id)
                return {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}, 'Invalid class ID returned by model.', None, None
        else:
            print("No objects detected in image")
            return {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}, 'No objects detected in the image.', None, None

        # Annotate image
        annotated_image = cv2.imread(image_path)  # Reload original image
        if detected_class and annotated_image is not None:
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            _, buffer = cv2.imencode('.jpg', annotated_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            print("Detection image generated")
        else:
            image_base64 = None
            print("Failed to generate detection image")

        percentages = {k: (v * 100) for k, v in class_counts.items()}
        dominant_class = max(class_counts, key=class_counts.get) if max(class_counts.values()) > 0 else 'None'
        suggestions = {
            'Fresh': 'The meat is fresh and safe for consumption. Store in a refrigerator at 0-4°C.',
            'Half-Fresh': 'The meat is partially spoiled. Consider cooking thoroughly before use.',
            'Spoiled': 'The meat is spoiled and unsafe. Dispose of it immediately.',
            'None': 'No objects detected. Please try another image.'
        }
        suggestion = suggestions.get(dominant_class, 'No suggestion available.')
        print(f"Image processing complete: percentages={percentages}, suggestion={suggestion}")
        
        return percentages, suggestion, image_base64, None
    except Exception as e:
        print("Error in process_image:", traceback.format_exc())
        return {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}, f'Image processing failed: {str(e)}', None, None

# Function to process video (full annotated video)
def process_video(video_path):
    try:
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video:", video_path)
            return {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}, 'Failed to open video.', None, None

        class_counts = {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}
        total_frames = 0
        annotated_frame_base64 = None
        video_base64 = None

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            # Process first frame for image output
            if total_frames == 0:
                model_input = cv2.resize(frame, (640, 640))
                results = model(model_input)
                print("First frame inference results:", results)
                detections = sv.Detections.from_ultralytics(results[0])
                if len(detections) > 0:
                    detections.xyxy[0] = shrink_box(detections.xyxy[0], scale=0.8)
                    annotated_frame = frame.copy()
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    print("Video frame annotated for image output")
                else:
                    print("No objects detected in first frame")
                    annotated_frame_base64 = None

            # Process frame for video output
            model_input = cv2.resize(frame, (640, 640))
            results = model(model_input)
            detections = sv.Detections.from_ultralytics(results[0])
            if len(detections) > 0:
                detections.xyxy[0] = shrink_box(detections.xyxy[0], scale=0.8)
                class_id = int(detections.class_id[0])
                if class_id < len(model.names):
                    class_name = model.names[class_id]
                    class_counts[class_name] += 1
                else:
                    print("Invalid class ID in frame:", class_id)
            else:
                print(f"No objects detected in frame {total_frames}")
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            out.write(annotated_frame)
            total_frames += 1

        cap.release()
        out.release()

        if total_frames == 0:
            print("No frames processed in video")
            if os.path.exists(out_path):
                os.remove(out_path)
            return {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}, 'No frames processed in video.', None, None

        # Encode video as base64
        if os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode('utf-8')
            os.remove(out_path)
            print("Annotated video generated")
        else:
            print("Failed to generate annotated video")
            video_base64 = None

        percentages = {k: (v / total_frames * 100) if total_frames > 0 else 0 for k, v in class_counts.items()}
        dominant_class = max(class_counts, key=class_counts.get) if max(class_counts.values()) > 0 else 'None'
        suggestions = {
            'Fresh': 'The meat is fresh and safe for consumption. Store in a refrigerator at 0-4°C.',
            'Half-Fresh': 'The meat is partially spoiled. Consider cooking thoroughly before use.',
            'Spoiled': 'The meat is spoiled and unsafe. Dispose of it immediately.',
            'None': 'No objects detected in the video. Please try another video.'
        }
        suggestion = suggestions.get(dominant_class, 'No suggestion available.')
        print(f"Video processing complete: percentages={percentages}, suggestion={suggestion}, annotated_frame={bool(annotated_frame_base64)}, video={bool(video_base64)}")

        return percentages, suggestion, annotated_frame_base64, video_base64
    except Exception as e:
        print("Error in process_video:", traceback.format_exc())
        if 'out' in locals():
            out.release()
        if 'out_path' in locals() and os.path.exists(out_path):
            os.remove(out_path)
        return {'Fresh': 0, 'Half-Fresh': 0, 'Spoiled': 0}, f'Video processing failed: {str(e)}', None, None

@app.route('/')
def index():
    if model is None:
        return jsonify({'error': 'Model failed to load. Check server logs.'}), 500
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Received upload request")
        if 'file' not in request.files:
            print("No file provided in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.mp4', '.avi'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            print(f"Invalid file type: {file_ext}")
            return jsonify({'error': 'Invalid file type. Use images (.png, .jpg, .jpeg, .bmp) or videos (.mp4, .avi)'}), 400

        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"File saved: {file_path}")

        # Process file
        is_image = file_ext in {'.png', '.jpg', '.jpeg', '.bmp'}
        print(f"Starting processing for {'image' if is_image else 'video'}")
        percentages, suggestion, image_base64, video_base64 = process_image(file_path) if is_image else process_video(file_path)
        
        response = jsonify({
            'percentages': percentages,
            'suggestion': suggestion,
            'image_base64': image_base64,
            'video_base64': video_base64
        })
        print("Returning response:", response.get_json())
        return response
    except Exception as e:
        print("Error in upload_file:", traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            print(f"File deleted: {file_path}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)  # Disable auto-reload