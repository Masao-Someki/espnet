# googlei18n_lowresource tts data recipe

This egs3 data module reads Kaldi `wav.scp` and `text` files prepared from
`egs2/googlei18n_lowresource/tts1/local/data.sh`.

Set `GOOGLEI18N` for the downloaded corpus and use the egs2 preparation flow,
or pass `source_dir` to the prepared Kaldi data directory.
