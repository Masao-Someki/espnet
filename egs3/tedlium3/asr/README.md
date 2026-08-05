# tedlium3 asr recipe

This egs3 data module reads the Kaldi `wav.scp` and `text` files produced by
the corresponding egs2 recipe: `egs2/tedlium3/asr1/local/data.sh`.

Set `TEDLIUM3` to the downloaded corpus location when using the egs2 data
preparation flow, or place the prepared `data/{train,dev,test}` directories
under this recipe. Split aliases are defined in `dataset/__init__.py`.
