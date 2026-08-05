# must_c st data recipe

This egs3 data module reads Kaldi `wav.scp` and `text` files prepared from
`egs2/must_c/st1/local/data.sh`.

Set `MUST_C` for the downloaded corpus and use the egs2 preparation flow, or
pass `source_dir` to the prepared Kaldi data directory. The default `test`
alias maps to `tst-COMMON`.
