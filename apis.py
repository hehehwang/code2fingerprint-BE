import json
from base64 import b64decode

import numpy as np

from fingerprinter import ScFp


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


def sourceCodeAnalysis_api(request):
    """
    req = {"source_code": source code encoded base64}
    """
    req = request.get_json()
    try:
        sourceCode = ScFp(b64decode(req["source_code"]).decode())
        output = {
            "original_source_code": sourceCode,
            "content": sourceCode.methods,
            "status": "success",
        }
        return json.dumps(output, cls=NumpyEncoder)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}), 400


def similarityBtwSourceCodes_api(request):
    """
    req = {
    "original_source_code": source code encoded base64
    "target_source_code": source code encoded base64
    "similarity_threshold": threshold to judge its method similar
    }
    """
    req = request.get_json()
    try:
        originalSc = b64decode(req["original_source_code"]).decode()
        targetSc = b64decode(req["target_source_code"]).decode()
        result = ScFp.findSimilarMethodsBtw(originalSc, targetSc)
        output = {
            "status": "success",
            "content": result,
            "length": len(result),
        }

        return json.dumps(output, cls=NumpyEncoder)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}), 400
