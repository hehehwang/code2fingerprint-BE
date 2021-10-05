import json
from typing import List, cast

import torch
from code2seq.data.vocabulary import Vocabulary
from flask import Flask, request
from flask_cors import CORS
from omegaconf import OmegaConf

from pyAstParser.astParser import PyASTParser
from getLabeledPathContext import PathContextConvert
from Code2Fingerprint import Code2Seq
from base64 import b64decode

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
MODEL = Code2Seq.load_from_checkpoint('checkpoint/epoch9.ckpt')


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


def convertSequence(sequence: list):
    converted = []
    for token in sequence:
        if token == '<SOS>':
            continue
        elif token == '<EOS>':
            break
        converted.append(token)

    return '_'.join(converted)


def runModel(pathContext: str):
    converted = converter.getPathContext(pathContext)
    from_token = torch.tensor(transpose([path.from_token for path in converted.path_contexts]), dtype=torch.long)
    path_nodes = torch.tensor(transpose([path.path_node for path in converted.path_contexts]), dtype=torch.long)
    to_token = torch.tensor(transpose([path.to_token for path in converted.path_contexts]), dtype=torch.long)
    contexts = torch.tensor([len(converted.path_contexts)])

    encoded, logit = MODEL(from_token=from_token,
                           path_nodes=path_nodes,
                           to_token=to_token,
                           contexts_per_label=contexts,
                           output_length=7)
    predictions = logit.squeeze(1).argmax(-1)
    labels = [id_to_label[i.item()] for i in predictions]

    return encoded, labels

@app.route('/')
def index():
    return 'hello!'

@app.route('/api', methods=['POST'])
def api():
    req = request.get_json()
    sourceCode = b64decode(req['source_code']).decode()
    # try:
    output = {'original_source_code': sourceCode, 'content': []}
    parser = PyASTParser()
    parsedCode = parser.readSourceCode(sourceCode)
    print(parsedCode)
    for data in parsedCode:
        fragment = {'function_name': data['method_name'], 'function_source_code': data['source_code']}
        encoded, labels = runModel(data['parsed_line'])
        fragment['finger_print'] = encoded.mean(axis=0).tolist()
        fragment['predicted'] = convertSequence(labels)
        output['content'].append(fragment)
    output['status'] =  'success'
    return json.dumps(output)

    # except Exception as e:
    #     return json.dumps({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(host='192.168.0.26', port=5689, debug=True)