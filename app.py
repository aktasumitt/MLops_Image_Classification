from flask import Flask, render_template, request, redirect, send_from_directory 
import os
from werkzeug.utils import secure_filename
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.config.configuration import Configuration

# Klasör ayarları
configuration = Configuration()
UPLOAD_FOLDER = configuration.prediction_config().predict_data_path

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home_page():
    upload_message = ""
    results = {}

    if request.method == 'POST':
        action = request.form.get("action")

        if action == "upload":
            if 'files' not in request.files:
                return redirect(request.url)
            files = request.files.getlist('files')
            for file in files:
                if file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            upload_message = "Images uploaded successfully!"

        elif action == "predict":
            prediction_pipeline = PredictionPipeline()
            results = prediction_pipeline.run_prediction_pipeline()

        elif action == "delete":
            files_to_delete = request.form.getlist("delete_files")
            for file in files_to_delete:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                if os.path.exists(file_path):
                    os.remove(file_path)

    images = sorted(os.listdir(UPLOAD_FOLDER))
    return render_template("index.html", images=images, results=results, upload_message=upload_message)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
