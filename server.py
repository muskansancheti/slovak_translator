import os
import uuid
import json
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from app import process_industrial_pdf, DICT_PATH, load_custom_translations

app = Flask(__name__)
# Allow frontend (localhost:3000) to connect
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# In-memory job storage
jobs = {}

def run_translation_job(job_id, input_path, output_path):
    def progress_callback(current, total, message):
        progress_pct = int((current / total) * 100)
        jobs[job_id]["progress"] = progress_pct
        jobs[job_id]["message"] = message

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Initializing translation models..."
        jobs[job_id]["progress"] = 5

        process_industrial_pdf(input_path, output_path, progress_callback=progress_callback)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Translation completed successfully!"
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error processing job {job_id}: {error_msg}")
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Translation failed: {error_msg}"
        jobs[job_id]["error"] = error_msg


@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Slovak Industrial PDF Translator Backend is running",
        "endpoints": {
            "translate": "/api/translate",
            "status": "/api/status/<job_id>",
            "download": "/api/download/<job_id>",
            "dictionary": "/api/dict"
        }
    })


@app.route("/api/translate", methods=["POST"])
def translate_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    job_id = str(uuid.uuid4())
    original_filename = file.filename
    safe_filename = f"{job_id}_{original_filename}"
    
    input_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"translated_{safe_filename}")
    
    file.save(input_path)
    
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "File uploaded, waiting in queue...",
        "original_name": original_filename,
        "original_file": input_path,
        "translated_file": output_path,
        "error": None
    }
    
    # Start translation in background
    thread = threading.Thread(target=run_translation_job, args=(job_id, input_path, output_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "pending"})


@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "original_name": job["original_name"],
        "error": job["error"]
    })


@app.route("/api/download/<job_id>", methods=["GET"])
def download_pdf(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        return jsonify({"error": "Job not completed or not found"}), 404
    
    return send_file(
        job["translated_file"],
        as_attachment=True,
        download_name=f"translated_{job['original_name']}"
    )


@app.route("/api/dict", methods=["GET"])
def get_dict():
    translations = load_custom_translations()
    return jsonify(translations)


@app.route("/api/dict", methods=["POST"])
def update_dict():
    try:
        new_dict = request.json
        if not isinstance(new_dict, dict):
            return jsonify({"error": "Invalid payload format, expected a dictionary"}), 400
        
        with open(DICT_PATH, "w", encoding="utf-8") as f:
            json.dump(new_dict, f, ensure_ascii=False, indent=2)
            
        return jsonify({"success": True, "message": "Dictionary updated successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to save dictionary: {str(e)}"}), 500


@app.route("/api/view/<pdf_type>/<job_id>", methods=["GET"])
def view_pdf(pdf_type, job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    if pdf_type == "original":
        filepath = job["original_file"]
    elif pdf_type == "translated":
        if job["status"] != "completed":
            return jsonify({"error": "Translated file not ready"}), 400
        filepath = job["translated_file"]
    else:
        return jsonify({"error": "Invalid type"}), 400

    return send_file(filepath, mimetype="application/pdf")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)