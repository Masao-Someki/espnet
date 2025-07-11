#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 William Chen
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import humanfriendly
import numpy as np
import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.cnn import CNNFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.ssl.espnet_model import ESPnetSSLModel
from espnet2.ssl.loss.abs_loss import AbsSSLLoss
from espnet2.ssl.loss.hubert import HuBERTLoss
from espnet2.ssl.utils.mask import Masking
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import HuBERTCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(wav2vec_cnn=CNNFrontend, default=DefaultFrontend),
    type_check=AbsFrontend,
    default="wav2vec_cnn",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(sinc=LightweightSincConvs, linear=LinearProjection),
    type_check=AbsPreEncoder,
    default="linear",
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=TransformerEncoder,
        e_branchformer=EBranchformerEncoder,
        conformer=ConformerEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)
loss_choices = ClassChoices(
    "loss",
    classes=dict(hubert=HuBERTLoss),
    type_check=AbsSSLLoss,
    default="hubert",
)
util_choices = ClassChoices(
    "util",
    classes=dict(mask=Masking),
    type_check=torch.nn.Module,
    default=None,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetSSLModel,
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)


class SSLTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )
        group.add_argument(
            "--collate_fn_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for collate_fn class.",
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--num_classes",
            type=int,
            default=None,
            help="The number of classes in hubert",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        group.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        parser.add_argument(
            "--window_size",
            type=int,
            default=None,
            help="weights for additional loss terms (not first one)",
        )
        parser.add_argument(
            "--window_shift",
            type=int,
            default=None,
            help="weights for additional loss terms (not first one)",
        )
        group.add_argument(
            "--loss",
            action=NestedDictAction,
            default=[
                {
                    "name": "hubert",
                    "conf": {},
                },
            ],
            help="The loss functions and their configurations.",
        )
        group.add_argument(
            "--util",
            action=NestedDictAction,
            default=[
                {
                    "name": "mask",
                    "conf": {},
                },
            ],
            help="Configurations for SSL helper utils.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:

        # default sampling rate is 16000
        fs = args.frontend_conf.get("fs", 16000)
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        sample_rate = fs / 1000

        if args.encoder_conf.get("extractor_conv_layer_config", None) is None:
            # corresponding to default conv extractor
            # refer to espnet2/asr/encoder/hubert_encoder.py
            reception_field = 400
            stride_field = 320
        else:
            stride_field, reception_field = 1, 1
            for conv_config in args.encoder_conf["extractor_conv_layer_config"][::-1]:
                _, kernel, stride = conv_config
                stride_field *= stride
                reception_field = stride * (reception_field - 1) + kernel

        window_size = reception_field / sample_rate
        window_shift = stride_field / sample_rate
        return HuBERTCollateFn(
            float_pad_value=0.0,
            int_pad_value=args.collate_fn_conf.get("int_pad_value", -1),
            label_downsampling=args.collate_fn_conf.get("label_downsampling", 1),
            pad=args.collate_fn_conf.get("pad", False),
            rand_crop=args.collate_fn_conf.get("rand_crop", True),
            crop_audio=not args.collect_stats
            and args.collate_fn_conf.get("crop_audio", True),
            window_size=args.collate_fn_conf.get("window_size", window_size),
            window_shift=args.collate_fn_conf.get("window_shift", window_shift),
            sample_rate=sample_rate,
            train=train,
            mix_speech=args.collate_fn_conf.get("mix_speech", False),
            reverb_speech=args.collate_fn_conf.get("reverb_speech", False),
            noise_scp=args.collate_fn_conf.get("noise_scp", "data/noise/wav.scp"),
            rir_scp=args.collate_fn_conf.get("rir_scp", "data/rirs/wav.scp"),
            noise_apply_prob=args.collate_fn_conf.get("noise_apply_prob", 1.0),
            rir_apply_prob=args.collate_fn_conf.get("rir_apply_prob", 1.0),
            noise_db_range=args.collate_fn_conf.get("noise_db_range", "-5_20"),
            dynamic_mixing_gain_db=args.collate_fn_conf.get(
                "dynamic_mixing_gain_db", 5.0
            ),
            dynamic_mixing_prob=args.collate_fn_conf.get("dynamic_mixing_prob", 0.1),
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=getattr(args, "rir_scp", None),
                rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
                noise_scp=getattr(args, "noise_scp", None),
                noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
                noise_db_range=getattr(args, "noise_db_range", "13_15"),
                short_noise_thres=getattr(args, "short_noise_thres", 0.5),
                speech_volume_normalize=getattr(args, "rir_scp", None),
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech",)
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    @typechecked
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("text",)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> Union[ESPnetSSLModel]:
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {**args.frontend_conf}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            if "input_size" not in args.preencoder_conf:
                preencoder = preencoder_class(
                    input_size=input_size, **args.preencoder_conf
                )
            else:
                preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(
            input_size=input_size,
            **args.encoder_conf,
        )

        # 5. Loss
        losses = []
        for loss_args in args.loss:
            loss_conf = loss_args.get("conf", {})
            loss_class = loss_choices.get_class(loss_args["name"])
            loss = loss_class(
                encoder_output_size=encoder.output_size(),
                **loss_conf,
            )
            losses.append(loss)

        util_attributes = set()
        required_inputs = set()
        for loss_func in losses:
            util_attributes.update(loss_func.util_attributes)
            required_inputs.update(loss_func.required_inputs)
        util_modules = torch.nn.ModuleDict()

        for attr in util_attributes:
            util_args = {}
            util_class = util_choices.get_class(attr)
            for conf_obj in args.util:
                if conf_obj["name"] == attr:
                    util_args = conf_obj["conf"]
            if attr == "ema":
                util_args["model"] = encoder
                util_args["device"] = f"cuda:{torch.cuda.current_device()}"
            if attr == "mask":
                util_args["encoder_embed_dim"] = encoder.output_size()
            util = util_class(**util_args)
            util_modules.update({attr: util})

        # 8. Build model
        model_class = model_choices.get_class(args.model)

        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            token_list=token_list,
            losses=losses,
            util_attributes=util_attributes,
            required_inputs=required_inputs,
            util_modules=util_modules,
            **args.model_conf,
        )

        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
