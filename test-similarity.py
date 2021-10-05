import re
import time
from typing import List, cast

import seaborn as sns
import torch
from code2seq.data.vocabulary import Vocabulary
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from getLabeledPathContext import PathContextConvert
from Code2Fingerprint import Code2Fingerprint
from pyAstParser.astParser import PyASTParser
import numpy as np

config = OmegaConf.load('config/0927.yaml')
VOCAB = Vocabulary('vocabulary.pkl',
                   max_labels=config.data['max_labels'],
                   max_tokens=config.data.max_tokens)
id_to_label = {idx: lab for (lab, idx) in VOCAB.label_to_id.items()}
converter = PathContextConvert(VOCAB, config.data, True)
c2f = Code2Fingerprint.load_from_checkpoint('checkpoint/epoch9.ckpt')


def getBatchedFpandPredictions(pathContexts: list):
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

    samples = [converter.getPathContext(pathContext) for pathContext in pathContexts]
    labels = torch.tensor(transpose([s.label for s in samples]), dtype=torch.long)
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
    batchedFingerprint, batchedLogit = c2f(from_token=from_token,
                             path_nodes=path_nodes,
                             to_token=to_token,
                             contexts_per_label=contexts_per_label,
                             output_length=7)
    predictions = batchedLogit.squeeze(1).argmax(-1).T
    sequences = [[id_to_label[p.item()] for p in prediction] for prediction in predictions ]
    predictedWords = [convertSequenceToWord(sequence) for sequence in sequences]
    fingerprints = []
    idx = 0
    for unBatchCount in contexts_per_label:
        fingerprint = batchedFingerprint[idx:idx+unBatchCount]
        fingerprints.append(fingerprint.detach().numpy().mean(axis=0))
        idx += unBatchCount

    return fingerprints, predictedWords


# def getFpAndPrediction(pathContext: str):
#     def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
#         return [cast(List[int], it) for it in zip(*list_of_lists)]
#
#     def convertSequenceToWord(sequence: list):
#         words = []
#         for token in sequence:
#             if token == '<SOS>':
#                 continue
#             elif token == '<EOS>':
#                 break
#             words.append(token)
#
#         return '_'.join(words)
#
#     converted = converter.getPathContext(pathContext)
#     from_token = torch.tensor(transpose([path.from_token for path in converted.path_contexts]), dtype=torch.long)
#     path_nodes = torch.tensor(transpose([path.path_node for path in converted.path_contexts]), dtype=torch.long)
#     to_token = torch.tensor(transpose([path.to_token for path in converted.path_contexts]), dtype=torch.long)
#     contexts = torch.tensor([len(converted.path_contexts)])
#
#     fingerprints, logits = [], []
#     for _ in range(5):
#         fingerprint, logit = c2f(from_token=from_token,
#                                  path_nodes=path_nodes,
#                                  to_token=to_token,
#                                  contexts_per_label=contexts,
#                                  output_length=7)
#         fingerprints.append(fingerprint), logits.append(logit)
#     fingerprint = sum(fingerprints) / len(fingerprints)
#     logit = sum(logits) / len(logits)
#
#     predictions = logit.squeeze(1).argmax(-1)
#     sequence = [id_to_label[i.item()] for i in predictions]
#     predictedWord = convertSequenceToWord(sequence)
#
#     return fingerprint.detach().numpy().mean(axis=0), predictedWord


def getParsedCode(sourceCode): # {'method_name', 'source_code', 'parsed_line'}
    parser = PyASTParser()
    parser.readSourceCode(sourceCode)
    return parser.methodsData


def getFpDifference(fingerPrintHere, fingerPrintThere):
    return sum([abs(i) for i in fingerPrintHere - fingerPrintThere])

SIMILAR_THRESHOLD = 80
# while 1:
#     if input('Edit sample1 & sample2.py and enter or type "n" to stop ') == 'n': break

with open('sample1.py', 'r', encoding='utf-8') as f:
    code1 = ''.join(f.readlines())

with open('sample2.py', 'r', encoding='utf-8') as f:
    code2 = ''.join(f.readlines())

originalMethods, targetMethods = getParsedCode(code1), getParsedCode(code2)
originalBatchedCode, targetBatchedCode = getBatchedFpandPredictions([method['parsed_line'] for method in originalMethods]),\
                                         getBatchedFpandPredictions([method['parsed_line'] for method in targetMethods])
for i, method in enumerate(originalMethods):
    originalMethods[i]['fingerprint'], originalMethods[i]['predicted'] = originalBatchedCode[0][i], originalBatchedCode[1][i]
for i, method in enumerate(targetMethods):
    targetMethods[i]['fingerprint'], targetMethods[i]['predicted'] = targetBatchedCode[0][i], targetBatchedCode[1][i]

similarMethods = []
for targetMethod in targetMethods:
    targetFingerprint = targetMethod['fingerprint']
    mostSimilarMethodFromOriginal, difference = '', 300
    for originalMethod in originalMethods:

        newDifference = getFpDifference(originalMethod['fingerprint'], targetFingerprint)
        if newDifference < difference:
            mostSimilarMethodFromOriginal, difference = originalMethod, newDifference

    if difference < SIMILAR_THRESHOLD:
        similarMethods.append({"original": mostSimilarMethodFromOriginal,
                               "target": targetMethod})

print(f'similar methods in my code: {len(similarMethods)}/{len(originalMethods)}')
print(f'similar methods in your code: {len(similarMethods)}/{len(targetMethods)}')

for method in similarMethods:
    print(f'similar method : {method["original"]["method_name"]}({method["original"]["predicted"]}),'
          f' {method["target"]["method_name"]}({method["target"]["predicted"]})')
    print(f'similarity: {getFpDifference(method["original"]["fingerprint"], method["target"]["fingerprint"])}')
# myPredictedMethodNames = {}
# myMethods = []
# similarMethods = []
# for data in myCode:  # {'method_name', 'source_code', 'parsed_line'}
#     data['fingerprint'], data['predicted'] = getFpAndPrediction(data['parsed_line'])
#     myMethods.append(data)
# for yourMethod in yourCode:
#     yourMethod['fingerprint'], yourMethod['predicted'] = getFpAndPrediction(yourMethod['parsed_line'])
#     yourFingerprint, yourPredicted = yourMethod['fingerprint'], yourMethod['predicted']
#
#     myMostSimilarMethod, difference = myMethods[0], getFpDifference(myMethods[0]['fingerprint'], yourFingerprint)
#     for myMethod in myMethods[1:]:
#         newDifference = getFpDifference(myMethod['fingerprint'], yourFingerprint)
#         if newDifference < difference:
#             myMostSimilarMethod = myMethod
#             difference = newDifference
#     if difference < SIMILAR_THRESHOLD:
#         similarMethods.append({"original": myMostSimilarMethod, "target":yourMethod})
#
#
# print(f'similar methods in my code: {len(similarMethods)}/{len(myCode)}')
# print(f'similar methods in your code: {len(similarMethods)}/{len(yourCode)}')
#
# for method in similarMethods:
#     print(f'similar method : {method["original"]["method_name"]} - {method["original"]["predicted"]},'
#           f' {method["target"]["method_name"]} - {method["target"]["predicted"]}')
#     print(f'similarity: {getFpDifference(method["original"]["fingerprint"], method["target"]["fingerprint"])}')





# for i in range(min(map(len, [parsedCode1, parsedCode2]))):
#     print(parsedCode1[i]['method_name'], parsedCode2[i]['method_name'])
#     print(parsedCode1[i]['predicted'], parsedCode2[i]['predicted'])
#     print('\n'.join(parsedCode1[i]['parsed_line'].split()))
#     print('\n'.join(parsedCode2[i]['parsed_line'].split()))
#     print(getFpDifference(parsedCode1[i]['fingerprint'], parsedCode2[i]['fingerprint']))
#     print()
