from flask import Flask, render_template, request, jsonify
import os
import logging
import numpy as np

from mlProject.pipeline.prediction import PredictionPipeline, ProductionPredictionPipeline
from mlProject.core.exceptions import handle_exception, PredictionException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "ml-pipeline-flask",
        "version": "1.0.0"
    })

@app.route('/train', methods=['GET'])
def training():
    try:
        logger.info("Starting training pipeline via Flask")
        os.system("python main.py --mode train --production")
        return jsonify({
            "status": "success",
            "message": "Training completed successfully"
        })
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        handle_exception(e, PredictionException)
        return jsonify({
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form.get('fixed_acidity', 0))
            volatile_acidity = float(request.form.get('volatile_acidity', 0))
            citric_acid = float(request.form.get('citric_acid', 0))
            residual_sugar = float(request.form.get('residual_sugar', 0))
            chlorides = float(request.form.get('chlorides', 0))
            free_sulfur_dioxide = float(request.form.get('free_sulfur_dioxide', 0))
            total_sulfur_dioxide = float(request.form.get('total_sulfur_dioxide', 0))
            density = float(request.form.get('density', 0))
            pH = float(request.form.get('pH', 0))
            sulphates = float(request.form.get('sulphates', 0))
            alcohol = float(request.form.get('alcohol', 0))

            logger.info(f"Prediction request received: fixed_acidity={fixed_acidity}")

            data = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                density, pH, sulphates, alcohol
            ]])

            try:
                obj = ProductionPredictionPipeline()
                prediction = obj.predict(data)
            except Exception:
                obj = PredictionPipeline()
                prediction = obj.predict(data)

            logger.info(f"Prediction result: {prediction}")

            return render_template('results.html', prediction=str(prediction[0]))

        except Exception as e:
            logger.exception(f'Prediction error: {e}')
            return render_template('error.html', error=str(e))

    else:
        return render_template("index.html")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Content-Type must be application/json"
            }), 400

        data = request.get_json()

        required_fields = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid',
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {missing_fields}"
            }), 400

        input_data = np.array([[
            data['fixed_acidity'], data['volatile_acidity'], data['citric_acid'],
            data['residual_sugar'], data['chlorides'], data['free_sulfur_dioxide'],
            data['total_sulfur_dioxide'], data['density'], data['pH'],
            data['sulphates'], data['alcohol']
        ]])

        pipeline = ProductionPredictionPipeline()
        prediction = pipeline.predict(input_data)

        return jsonify({
            "status": "success",
            "prediction": float(prediction[0]),
            "model_info": pipeline.model_info
        })

    except Exception as e:
        logger.exception(f"API prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def api_predict_batch():
    try:
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Content-Type must be application/json"
            }), 400

        data = request.get_json()

        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({
                "status": "error",
                "message": "Invalid request: 'data' field must be an array of samples"
            }), 400

        required_count = 11
        for i, sample in enumerate(data['data']):
            if not isinstance(sample, list) or len(sample) != required_count:
                return jsonify({
                    "status": "error",
                    "message": f"Sample {i} must have exactly {required_count} features"
                }), 400

        input_data = np.array(data['data'])

        pipeline = ProductionPredictionPipeline()
        result = pipeline.predict_batch(input_data)

        return jsonify({
            "status": "success",
            "predictions": result['predictions'],
            "count": result['count'],
            "inference_time_ms": result['inference_time_ms']
        })

    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception(f"Server error: {e}")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500


if __name__ == "__main__":
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    if debug_mode:
        logger.info("Starting Flask app in DEBUG mode")
        app.run(host="0.0.0.0", port=8080, debug=True)
    else:
        logger.info("Starting Flask app in PRODUCTION mode")
        app.run(host="0.0.0.0", port=8080, threaded=True)