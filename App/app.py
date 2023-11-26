from flask import Flask, request, render_template
from StockLSTM import StockLSTM
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Set up a folder for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the uploaded file from the request
        uploaded_file = request.files["file"]

        # Check if the file is present and has a valid extension
        if uploaded_file and uploaded_file.filename.endswith(".json"):

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            uploaded_file.save(file_path)

            # Read the CSV file into a DataFrame
            lstm_model = StockLSTM(batch_size=16, filepath=file_path)
            lstm_model.train_model(epochs=50)
            pred = lstm_model.make_prediction()

            return render_template("index.html", output_result=pred)

        else:
            return render_template("index.html", error="Invalid file format. Please upload a JSON file.")

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
