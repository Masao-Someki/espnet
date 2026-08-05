# wenetspeech asr recipe

This egs3 data module reads the Kaldi `wav.scp` and `text` files produced by
`egs2/wenetspeech/asr1/local/data.sh`.

Set `WENETSPEECH` to the downloaded corpus location when using the egs2 data
preparation flow, or place prepared `data/{train,dev,test}` directories under
this recipe.
