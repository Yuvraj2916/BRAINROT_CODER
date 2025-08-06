# BRAINROT_CODER
Here we trained an ai model for object detection.
YOLO Detector Web App
A modern, interactive web application for running and visualizing results of your custom YOLO (You Only Look Once) object detection model (ONNX).
Supports drag-and-drop detection, real-time chart metrics, and a responsive, visually appealing UI.

✨ Features
Drag & Drop or Browse: Easily test images using the drop zone UI.

ONNX Model Inference in Browser: Fast, local inference using ONNX Runtime Web.

Visualization: Bounding boxes, class labels, and confidence for detections.

Training Metrics Dashboard: Beautiful, interactive charts for model evaluation.

Responsive Design: Works well on desktop and mobile.

Live Analysis: Switch between detection and metric analysis tabs.

🚀 Getting Started
1. Prerequisites
Node.js or Python 3 (for running a local server)

Modern browser (Chrome, Edge, Firefox, Safari)

VS Code (recommended but not required)

2. Files & Structure
Place these files in one directory (example structure):

text
yolo-webapp/
├── index.html
├── style.css
├── app.js
├── best1.onnx   # <--- Your YOLO ONNX model
3. Install (if needed)
No installation needed—everything runs in your browser!

4. Start a Local Web Server
Why? Browsers block file access for ONNX models unless served via HTTP.

Option 1: Python (cross-platform)
Open your terminal in your project folder:

bash
python3 -m http.server 8000
Go to http://localhost:8000 in your browser.

Option 2: Node.js (if installed)
bash
npm install -g http-server
http-server .
Visit http://localhost:8080 (or the port displayed).

Option 3: VS Code "Live Server" Extension (easy for Windows)
Install the Live Server extension.

Right-click index.html → "Open with Live Server"

🖼️ Usage
Open the App:
Go to the local URL in your browser (see above).

Detect Objects:

On the "Detector" tab, drag an image into the drop zone or click 'browse' and select a file.

See detected objects overlaid on your image.

Analyze Metrics:

Switch to the "Analysis" tab for a dashboard of training/evaluation charts and insights.

🎨 Customization
To Use Your Model:
Replace best1.onnx with your own ONNX model file (ensure the filename and class list in app.js match your model).

Edit Appearance:
Adjust style.css for color/theme, or update HTML for new features.

🛠️ Built With
ONNX Runtime Web

Chart.js

three.js (for dynamic backgrounds)

Vanilla JavaScript + CSS

⚠️ Troubleshooting
Model not loading?

Double-check that best1.onnx is in the same folder as index.html.

You must use a local server—don't double-click HTML.

UI/CSS issues?

Make sure style.css is referenced correctly in index.html.

Detection incorrect?

Update the class list and input pre/post-processing code as needed for your ONNX model
