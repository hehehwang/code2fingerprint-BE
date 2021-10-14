import json
from typing import List, cast

import torch
from code2seq.data.vocabulary import Vocabulary
from flask import Flask, request
from flask_cors import CORS
from omegaconf import OmegaConf

from pyAstParser.astParser import PyASTParser
from getLabeledPathContext import PathContextConvert
from Code2Fingerprint import Code2Fingerprint
from base64 import b64decode
import numpy as np

app = Flask(__name__)
cors = CORS(app, resources={
    r"/api/*": {"origin": "*"},
})

MODEL_CONFIG = OmegaConf.load('config/0927.yaml')
VOCAB = Vocabulary('vocabulary.pkl',
                   max_labels=MODEL_CONFIG.data['max_labels'],
                   max_tokens=MODEL_CONFIG.data.max_tokens)
id_to_label = {idx: lab for (lab, idx) in VOCAB.label_to_id.items()}
converter = PathContextConvert(VOCAB, MODEL_CONFIG.data, True)
MODEL = Code2Fingerprint.load_from_checkpoint('checkpoint/epoch9.ckpt')


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


def getParsedCode(sourceCode):  # {'method_name', 'source_code', 'parsed_line'}
    parser = PyASTParser()
    parser.readSourceCode(sourceCode)
    return parser.methodsData


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


def convertSequenceToWord(sequence: list):
    converted = []
    for token in sequence:
        if token == '<SOS>':
            continue
        elif token == '<EOS>':
            break
        converted.append(token)

    return '_'.join(converted)


def getBatchedFpandPredictions(batchedPathContexts: list):
    samples = [converter.getPathContext(pathContext) for pathContext in batchedPathContexts]
    contexts_per_label = torch.tensor([len(s.path_contexts) for s in samples])
    from_token = torch.tensor(
        transpose([path.from_token for s in samples for path in s.path_contexts]), dtype=torch.long
    )
    path_nodes = torch.tensor(
        transpose([path.path_node for s in samples for path in s.path_contexts]), dtype=torch.long
    )
    to_token = torch.tensor(
        transpose([path.to_token for s in samples for path in s.path_contexts]), dtype=torch.long
    )
    batchedFingerprint, batchedLogit = MODEL(from_token=from_token,
                                             path_nodes=path_nodes,
                                             to_token=to_token,
                                             contexts_per_label=contexts_per_label,
                                             output_length=7)
    predictions = batchedLogit.squeeze(1).argmax(-1).T
    if len(predictions.size()) != 1:
        sequences = [[id_to_label[p.item()] for p in prediction] for prediction in predictions]
    else:
        sequences = [[id_to_label[p.item()] for p in predictions]]
    predictedWords = [convertSequenceToWord(sequence) for sequence in sequences]
    fingerprints = []
    idx = 0
    for unBatchCount in contexts_per_label:
        fingerprint = batchedFingerprint[idx:idx + unBatchCount]
        fingerprints.append(fingerprint.detach().numpy().mean(axis=0))
        idx += unBatchCount

    return fingerprints, predictedWords


def getAnalysedMethodsFromSourceCode(sourceCode: str):
    parsedMethods = getParsedCode(sourceCode)
    batchedAnalysedMethods = getBatchedFpandPredictions([method['parsed_line'] for method in parsedMethods])
    for i, method in enumerate(parsedMethods):
        parsedMethods[i]['fingerprint'], parsedMethods[i]['predicted'] = batchedAnalysedMethods[0][i], \
                                                                         batchedAnalysedMethods[1][i]
    return parsedMethods


def getFpDifference(fingerPrintHere, fingerPrintThere):
    return sum([abs(i) for i in fingerPrintHere - fingerPrintThere])


@app.route('/')
def index():
    return 'hello!'


@app.route('/api/sourceCodeAnalysis', methods=['POST'])
def sourceCodeAnalysis():
    """
    req = {"source_code": source code encoded base64}
    """
    req = request.get_json()
    # try:
    sourceCode = b64decode(req['source_code']).decode()
    output = {'original_source_code': sourceCode,
              'content': getAnalysedMethodsFromSourceCode(sourceCode),
              'status': 'success'
              }
    return json.dumps(output, cls=NumpyEncoder)

    # except Exception as e:
    #     return json.dumps({"status": "error", "message": str(e)})


@app.route('/api/similarity', methods=['POST'])
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
        similarityThreshold = int(req['similarity_threshold'])
        originalSourceCode = b64decode(req['original_source_code']).decode()
        targetSourceCode = b64decode(req['target_source_code']).decode()
        originalMethods = getAnalysedMethodsFromSourceCode(originalSourceCode)
        targetMethods = getAnalysedMethodsFromSourceCode(targetSourceCode)
        similarMethods = []
        for targetMethod in targetMethods:
            targetFingerprint = targetMethod['fingerprint']
            mostSimilarMethodFromOriginal, difference = originalMethods[0], 300
            for originalMethod in originalMethods:

                newDifference = getFpDifference(originalMethod['fingerprint'], targetFingerprint)
                if newDifference < difference:
                    mostSimilarMethodFromOriginal, difference = originalMethod, newDifference

            if difference < similarityThreshold:
                similarMethods.append({"original": mostSimilarMethodFromOriginal['method_name'],
                                       "target": targetMethod['method_name'],
                                       "difference": round(difference, 2)})
        output = {"status": "success", "content": similarMethods, "length": len(similarMethods)}

        return json.dumps(output, cls=NumpyEncoder)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5689, debug=True)
