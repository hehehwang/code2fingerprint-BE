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

config = OmegaConf.load('config/0927.yaml')
VOCAB = Vocabulary('vocabulary.pkl',
                   max_labels=config.data['max_labels'],
                   max_tokens=config.data.max_tokens)
id_to_label = {idx: lab for (lab, idx) in VOCAB.label_to_id.items()}
converter = PathContextConvert(VOCAB, config.data, True)
c2f = Code2Fingerprint.load_from_checkpoint('checkpoint/epoch9.ckpt')


def getOutput(pathContext: str):
    def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
        return [cast(List[int], it) for it in zip(*list_of_lists)]

    converted = converter.getPathContext(pathContext)
    from_token = torch.tensor(transpose([path.from_token for path in converted.path_contexts]), dtype=torch.long)
    path_nodes = torch.tensor(transpose([path.path_node for path in converted.path_contexts]), dtype=torch.long)
    to_token = torch.tensor(transpose([path.to_token for path in converted.path_contexts]), dtype=torch.long)
    contexts = torch.tensor([len(converted.path_contexts)])

    encoded, logit = c2f(from_token=from_token,
                         path_nodes=path_nodes,
                         to_token=to_token,
                         contexts_per_label=contexts,
                         output_length=7)
    predictions = logit.squeeze(1).argmax(-1)
    labels = [id_to_label[i.item()] for i in predictions]

    return encoded, labels


def convertSequence(sequence: list):
    converted = []
    for token in sequence:
        if token == '<SOS>':
            continue
        elif token == '<EOS>':
            break
        converted.append(token)

    return '_'.join(converted)


def getTimeStamp() -> int:
    return int(time.time())


def delimFileName(fileName: str) -> str:
    return re.sub('[\/:*?"<>|]', '', fileName)


while 1:
    iii = input('Edit sample.py and enter or type "n" to stop ')
    if iii == 'n': break
    # testCode = ''
    with open('sample.py', 'r', encoding='utf-8') as f:
        testCode = '\n'.join(f.readlines())

    parser = PyASTParser()
    parser.readSourceCode(testCode)

    parsed = parser.methodsData

    for data in parsed:
        encoded, labels = getOutput(data['parsed_line'])
        data['encoded'] = encoded
        data['finger_print'] = encoded.mean(axis=0).detach().numpy()
        data['predicted'] = convertSequence(labels)

    for data in parsed:
        print(data['method_name'])
        fig = plt.figure()
        ax = sns.heatmap(data['finger_print'].reshape(16, 20), vmin=-1, vmax=1)
        plt.savefig(
            'output/' + data['method_name'] + '_' + delimFileName(data['predicted']) + '_' + str(getTimeStamp()))
        plt.close(fig)
