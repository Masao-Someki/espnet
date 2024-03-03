# The learn_kmeans.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/dump_km_label.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert


import argparse
import logging
import os
import sys
import random

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from ssl_feature_utils import (
    ESPnetHubertFeatureReader,
    HubertFeatureReader,
    MfccFeatureReader,
    S3PRLFeatureReader,
    build_data_iterator,
    format_feature_conf_str,
)

from espnet2.utils.types import str2bool
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_writers import file_writer_helper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


feature_reader_choice = dict(
    mfcc=MfccFeatureReader,
    fairseq_hubert=HubertFeatureReader,
    espnet_hubert=ESPnetHubertFeatureReader,
    s3prl=S3PRLFeatureReader,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--n_bits", type=int, default=10)
    parser.add_argument("--w", type=int, default=1)
    parser.add_argument("--lsh_algorithm", type=str, default='simple')
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--online_feature_extract", type=str2bool, default=False)
    parser.add_argument("--feature_conf", type=str, default=None)
    parser.add_argument("--batch_bins", type=int, default=1)
    parser.add_argument(
        "--utt2num_samples",
        type=str,
        default=None,
        help="Specify the utt2num_samples file.",
    )

    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound", "kaldi_ark"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--out_filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt"
    )

    return parser


class ApplyRP(object):
    def __init__(self, n_bits=10, w=1, lsh_algorithm='simple', use_gpu=True):
        self.lsh_algorithm = lsh_algorithm
        self.feat_dim = None
        self.n_bits = n_bits
        self.w = w
    
    def __call__(self, x):
        if self.lsh_algorithm == 'simple':
            return self.random_proj(x)
        elif self.lsh_algorithm == 'e2lsh':
            return self.e2lsh(x)
        else:
            raise ValueError(f'Unrecognized lsh_algorithm {self.lsh_algorithm}')
        
    def init_random_array(self, feat_dim):
        if self.lsh_algorithm == 'simple':
            self.random_matrix = torch.randn(1, feat_dim, self.n_bits)
            self.random_matrix_shape = (feat_dim, self.n_bits)
            self.bit_arange = 1 << torch.arange(self.n_bits)

        elif self.lsh_algorithm == 'e2lsh': # You cannot define the number of clusters.
            self.a = torch.randn(feat_dim) # normal dist
            self.b = torch.rand((1,)) * self.w # uniform dist from 0 to w

    def random_proj(self, x):
        """LSH with simple random projection.

        Args:
            x: (L, D)
        
        Returns:
            (L,)
        """
        if self.feat_dim is None:
            self.init_random_array(x.shape[-1])
        
        feat_len, _ = x.shape
        x = F.normalize(x, p=1.0, dim=1)
        inner_products = torch.einsum(
            'ld,ldr->lr', x, self.random_matrix.expand(feat_len, *self.random_matrix_shape)
        )
        buckets = (inner_products > 0).type(torch.int)
        buckets = (buckets * self.bit_arange).sum(-1)
        return buckets.numpy()

    def e2lsh(self, x):
        """E2LSH based on https://www.mit.edu/~andoni/LSH/

        Args:
            x: (L, D)
        
        Returns:
            (L,)
        """
        if self.feat_dim is None:
            self.init_random_array(x.shape[-1])

        feat_len, feat_size = x.shape
        return torch.floor((x @ self.a + self.b) / self.w).numpy()


def dump_label(
    rspecifier,
    in_filetype,
    wspecifier,
    out_filetype,
    n_bits,
    w,
    lsh_algorithm,
    use_gpu,
    online_feature_extract,
    **kwargs
):
    if online_feature_extract:
        assert "feature_conf" in kwargs
        # need to wrap arguments with double-quotes for json string
        feature_conf = format_feature_conf_str(kwargs["feature_conf"])
    else:
        feature_conf = None

    apply_random_proj = ApplyRP(n_bits, w, lsh_algorithm, use_gpu=use_gpu)

    if not online_feature_extract:
        # dumped ssl feature in kaldi ark format
        with file_writer_helper(
            wspecifier,
            filetype=out_filetype,
        ) as writer:
            for utt, feat in file_reader_helper(rspecifier, in_filetype):
                lab = apply_random_proj(feat)
                writer[utt] = lab
    else:
        assert feature_conf["type"] in feature_reader_choice
        reader_class = feature_reader_choice[feature_conf["type"]]
        reader_conf = feature_conf.get("conf", dict())

        if reader_conf.get("multilayer_feature", None):
            reader_conf["multilayer_feature"] = str2bool(
                reader_conf["multilayer_feature"]
            )
        if reader_conf.get("layer", None):
            reader_conf["layer"] = int(reader_conf["layer"])

        reader = reader_class(**reader_conf)
        iterator = build_data_iterator(
            rspecifier,
            in_filetype,
            utt2num_samples=args.utt2num_samples,
            batch_bins=kwargs.get("batch_bins", 1),
        )
        with file_writer_helper(
            wspecifier,
            filetype=out_filetype,
        ) as writer:
            for utt_ids, data in iterator:
                feats, feats_lens = reader.get_feats(
                    data["speech"], data["speech_lengths"]
                )

                for idx, utt in enumerate(utt_ids):
                    # lab = apply_random_proj(feats[idx][: feats_lens[idx]].numpy())
                    lab = apply_random_proj(feats[idx][: feats_lens[idx]])
                    writer[utt] = lab

    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    # Set seed
    # Python random
    random.seed(args.seed)
    # Numpy
    np.random.seed(args.seed)
    # Pytorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dump_label(**vars(args))
