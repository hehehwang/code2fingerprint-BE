from code2seq.model.code2seq import Code2Seq


class Code2Fingerprint(Code2Seq):
    def __init__(
        self,
        model_config,
        optimizer_config,
        vocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__(model_config, optimizer_config, vocabulary, teacher_forcing)

    def forward(
        self,
        from_token,
        path_nodes,
        to_token,
        contexts_per_label,
        output_length,
        target_sequence=None,
    ):
        encoded_paths = self._encoder(from_token, path_nodes, to_token)
        output_logits = self._decoder(
            encoded_paths, contexts_per_label, output_length, target_sequence
        )
        return encoded_paths, output_logits
