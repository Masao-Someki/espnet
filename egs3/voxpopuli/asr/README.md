# voxpopuli asr recipe

This egs3 data module reads the Kaldi `wav.scp` and `text` files produced by
`egs2/slue-voxpopuli/asr1/local/data.sh`.

Set `VOXPOPULI` to the downloaded corpus location when using the egs2 data
preparation flow, or place prepared `data/{train,devel,test}` directories under
this recipe. The `dev` split maps to egs2's `devel` directory.
