import json
from base64 import b64decode

import numpy as np
from flask import Flask, request
from flask_cors import CORS

from fingerprinter import getAnalysedMethodsFromSC, getFpDifference

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origin": "*"},})


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route("/")
def index():
    return "hello!"


@app.route("/api/sourceCodeAnalysis", methods=["POST"])
def sourceCodeAnalysis():
    """
    req = {"source_code": source code encoded base64}
    """
    req = request.get_json()
    try:
        sourceCode = b64decode(req["source_code"]).decode()
        output = {
            "original_source_code": sourceCode,
            "content": getAnalysedMethodsFromSC(sourceCode),
            "status": "success",
        }
        return json.dumps(output, cls=NumpyEncoder)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@app.route("/api/similarity", methods=["POST"])
def similarity():
    """
    req = {
    "original_source_code": source code encoded base64
    "target_source_code": source code encoded base64
    "similarity_threshold": threshold to judge its method similar
    }
    """
    req = request.get_json()
    try:
        similarityThreshold = int(req["similarity_threshold"])
        originalSourceCode = b64decode(req["original_source_code"]).decode()
        targetSourceCode = b64decode(req["target_source_code"]).decode()
        originalMethods = getAnalysedMethodsFromSC(originalSourceCode)
        targetMethods = getAnalysedMethodsFromSC(targetSourceCode)
        similarMethods = []
        for targetMethod in targetMethods:
            targetFingerprint = targetMethod["fingerprint"]
            mostSimilarMethodFromOriginal, difference = originalMethods[0], 300
            for originalMethod in originalMethods:

                newDifference = getFpDifference(
                    originalMethod["fingerprint"], targetFingerprint
                )
                if newDifference < difference:
                    mostSimilarMethodFromOriginal, difference = (
                        originalMethod,
                        newDifference,
                    )

            if difference < similarityThreshold:
                similarMethods.append(
                    {
                        "original": mostSimilarMethodFromOriginal["method_name"],
                        "target": targetMethod["method_name"],
                        "difference": round(difference, 2),
                    }
                )
        output = {
            "status": "success",
            "content": similarMethods,
            "length": len(similarMethods),
        }

        return json.dumps(output, cls=NumpyEncoder)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5689, debug=True)
