"""
codecs - Wrappers pour les codecs video
========================================

- base: Interface abstraite CodecBase
- ffmpeg_wrapper: H.264/H.265/VP9/AV1 via FFmpeg
- lmd_wrapper: Wrapper LMD-PPV
- codec_factory: Factory pattern pour creation

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from .base import CodecBase, EncodeResult, DecodeResult
from .ffmpeg_wrapper import FFmpegCodec, H264Codec, H265Codec, VP9Codec, AV1Codec
from .lmd_wrapper import LMDCodec
from .codec_factory import CodecFactory

__all__ = [
    'CodecBase', 'EncodeResult', 'DecodeResult',
    'FFmpegCodec', 'H264Codec', 'H265Codec', 'VP9Codec', 'AV1Codec',
    'LMDCodec', 'CodecFactory'
]
