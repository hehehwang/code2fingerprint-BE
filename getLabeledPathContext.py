from random import shuffle
from typing import Dict, List, Optional

from code2seq.data.path_context import LabeledPathContext, Path
from code2seq.data.vocabulary import Vocabulary
from omegaconf import DictConfig


class PathContextConvert:
    _separator = "|"

    def __init__(self, vocab, config: DictConfig, random_context: bool):
        self._config = config
        self._vocab = vocab
        self._random_context = random_context

    def getPathContext(self, sourceCode: str):
        raw_sample = sourceCode
        raw_label, *raw_path_contexts = raw_sample.split()
        n_contexts = min(len(raw_path_contexts), self._config.max_context)
        if self._random_context:
            shuffle(raw_path_contexts)
        raw_path_contexts = raw_path_contexts[:n_contexts]

        if self._config.max_label_parts == 1:
            label = self.tokenize_class(raw_label, self._vocab.label_to_id)
        else:
            label = self.tokenize_label(raw_label, self._vocab.label_to_id, self._config.max_label_parts)
        paths = [self._get_path(raw_path.split(",")) for raw_path in raw_path_contexts]
        return LabeledPathContext(label, paths)

    @staticmethod
    def tokenize_class(raw_class: str, vocab: Dict[str, int]) -> List[int]:
        return [vocab[raw_class]]

    @staticmethod
    def tokenize_label(raw_label: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        sublabels = raw_label.split(PathContextConvert._separator)
        max_parts = max_parts or len(sublabels)
        label_unk = vocab[Vocabulary.UNK]

        label = [vocab[Vocabulary.SOS]] + [vocab.get(st, label_unk) for st in sublabels[:max_parts]]
        if len(sublabels) < max_parts:
            label.append(vocab[Vocabulary.EOS])
            label += [vocab[Vocabulary.PAD]] * (max_parts + 1 - len(label))
        return label

    @staticmethod
    def tokenize_token(token: str, vocab: Dict[str, int], max_parts: Optional[int]) -> List[int]:
        sub_tokens = token.split(PathContextConvert._separator)
        max_parts = max_parts or len(sub_tokens)
        token_unk = vocab[Vocabulary.UNK]

        result = [vocab.get(st, token_unk) for st in sub_tokens[:max_parts]]
        result += [vocab[Vocabulary.PAD]] * (max_parts - len(result))
        return result

    def _get_path(self, raw_path: List[str]) -> Path:
        return Path(
            from_token=self.tokenize_token(raw_path[0], self._vocab.token_to_id, self._config.max_token_parts),
            path_node=self.tokenize_token(raw_path[1], self._vocab.node_to_id, self._config.path_length),
            to_token=self.tokenize_token(raw_path[2], self._vocab.token_to_id, self._config.max_token_parts),
        )
