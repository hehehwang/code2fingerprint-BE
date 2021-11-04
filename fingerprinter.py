import json
from dataclasses import dataclass
from typing import List, NamedTuple, Tuple, cast

import numpy as np
import torch
from code2seq.data.vocabulary import Vocabulary
from omegaconf import OmegaConf
from torch.functional import Tensor

from Code2Fingerprint import Code2Fingerprint
from getLabeledPathContext import PathContextConvert
from pyAstParser.astParser import PyASTParser

MODEL_CONFIG = OmegaConf.load("config/0927.yaml")
VOCAB = Vocabulary(
    "modelData/vocabulary.pkl",
    max_labels=MODEL_CONFIG.data["max_labels"],
    max_tokens=MODEL_CONFIG.data.max_tokens,
)
id_to_label = {idx: lab for (lab, idx) in VOCAB.label_to_id.items()}
PCC = PathContextConvert(VOCAB, MODEL_CONFIG.data, True)
MODEL = Code2Fingerprint.load_from_checkpoint("modelData/epoch9.ckpt")


@dataclass
class ParsedMethod:
    methodName: str
    sourceCode: str
    parsedLine: List[int]
    fingerprint: np.ndarray
    predicted: str


def getParsedMethodInSc(sourceCode: str) -> Tuple[ParsedMethod]:
    parser = PyASTParser()
    parser.readSourceCode(sourceCode)
    parsed = parser.methodsData
    psc = []
    for p in parsed:
        psc.append(
            ParsedMethod(
                p["method_name"], p["source_code"], p["parsed_line"], None, None
            )
        )
    psc = tuple(psc)
    return psc


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


def convertSequenceToWord(sequence: list):
    converted = []
    for token in sequence:
        if token == "<SOS>":
            continue
        elif token == "<EOS>":
            break
        converted.append(token)

    return "_".join(converted)


def getBatchedFpandPredictions(batchedPathContexts: list) -> Tuple[list, list]:
    if not batchedPathContexts:
        return [], []
    samples = [PCC.getPathContext(pathContext) for pathContext in batchedPathContexts]
    contexts_per_label = torch.tensor([len(s.path_contexts) for s in samples])
    from_token = torch.tensor(
        transpose([path.from_token for s in samples for path in s.path_contexts]),
        dtype=torch.long,
    )
    path_nodes = torch.tensor(
        transpose([path.path_node for s in samples for path in s.path_contexts]),
        dtype=torch.long,
    )
    to_token = torch.tensor(
        transpose([path.to_token for s in samples for path in s.path_contexts]),
        dtype=torch.long,
    )
    batchedFingerprint, batchedLogit = MODEL(
        from_token=from_token,
        path_nodes=path_nodes,
        to_token=to_token,
        contexts_per_label=contexts_per_label,
        output_length=7,
    )
    predictions = batchedLogit.squeeze(1).argmax(-1).T
    if len(predictions.size()) != 1:
        sequences = [
            [id_to_label[p.item()] for p in prediction] for prediction in predictions
        ]
    else:
        sequences = [[id_to_label[p.item()] for p in predictions]]
    predictedWords = [convertSequenceToWord(sequence) for sequence in sequences]
    fingerprints = []
    idx = 0
    for unBatchCount in contexts_per_label:
        fingerprint = batchedFingerprint[idx : idx + unBatchCount]
        fingerprints.append(fingerprint.detach().numpy().mean(axis=0))
        idx += unBatchCount

    return fingerprints, predictedWords


def getFpDifference(fingerPrintHere, fingerPrintThere):
    # return sum([abs(i) for i in fingerPrintHere - fingerPrintThere])
    return sum(abs(fingerPrintThere - fingerPrintHere))


class ScFp:
    SIM_THRESHOLD = 45

    def __init__(self, sourceCode: str):
        self.sourceCode = sourceCode
        self.methods: Tuple[ParsedMethod] = getParsedMethodInSc(sourceCode)
        self.getMethodsAnalysed()

    def getMethodsAnalysed(self) -> None:
        batchedAnalysedMethods = getBatchedFpandPredictions(
            [m.parsedLine for m in self.methods]
        )
        for i in range(len(self.methods)):
            self.methods[i].fingerprint, self.methods[i].predicted = (
                batchedAnalysedMethods[0][i],
                batchedAnalysedMethods[1][i],
            )
        return

    @classmethod
    def findSimilarMethodsBtw(
        cls, originalSourceCode: str, targetSourceCode: str
    ) -> List[dict]:
        originalSc = cls(originalSourceCode)
        targetSc = cls(targetSourceCode)
        similarMethods = []
        for targetMethod in targetSc.methods:
            targetFp = targetMethod.fingerprint
            difference, mostSimilarMethodFromOriginal = min(
                [
                    (getFpDifference(m.fingerprint, targetFp), m)
                    for m in originalSc.methods
                ]
            )

            if difference < cls.SIM_THRESHOLD:
                similarMethods.append(
                    {
                        "original": mostSimilarMethodFromOriginal.methodName,
                        "target": targetMethod.methodName,
                        "difference": round(difference, 2),
                    }
                )
        return similarMethods


def main():
    sc = ScFp(
        """def tokenize(twitter, sent):
    return twitter.morphs(sent)"""
    )
    print(sc.methods[0].fingerprint)
    print(type(sc.methods[0].fingerprint))


if __name__ == "__main__":
    main()
