from flask import Flask, request
from flask_cors import CORS

from apis import *

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origin": "*"},})


@app.route("/")
def index():
    return "hello!"


@app.route("/api/sourceCodeAnalysis", methods=["POST"])
def sourceCodeAnalysis():
    """
    req = {"source_code": source code encoded base64}
    """
    return sourceCodeAnalysis_api(request)


@app.route("/api/similarityBtwSourceCodes", methods=["POST"])
def similarityBtwSourceCodes():
    """
    req = {
    "original_source_code": source code encoded base64
    "target_source_code": source code encoded base64
    }
    """
    return similarityBtwSourceCodes_api(request)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5689, debug=True)
