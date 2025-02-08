from flask import Flask, render_template, request, jsonify
from ice_breaker_v6 import ice_break_with
import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('flask.log', maxBytes=1000000, backupCount=1),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.logger.info('Starting Flask application')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        app.logger.info('Processing new request')
        name = request.form.get("name")
        if not name:
            app.logger.warning('No name provided in request')
            return jsonify({"error": "Name is required"}), 400
        
        app.logger.info(f'Processing request for name: {name}')
        summary, profile_pic_url = ice_break_with(name=name)
        
        if not profile_pic_url or profile_pic_url == "/null":
            app.logger.warning('No valid profile picture URL found')
            return jsonify({
                "summary_and_facts": summary.to_dict(),
                "picture_url": None 
            })
        
        app.logger.info('Successfully processed request')
        return jsonify({
            "summary_and_facts": summary.to_dict(),
            "picture_url": profile_pic_url
        })
    except Exception as e:
        app.logger.error(f'Error processing request: {str(e)}', exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)