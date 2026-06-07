"""
codec_factory.py - Factory pour creation des codecs
=====================================================

Factory pattern pour creer les instances de codecs.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from typing import Dict, List, Optional, Type

from .base import CodecBase
from .ffmpeg_wrapper import H264Codec, H265Codec, VP9Codec, AV1Codec, FFmpegCodec
from .lmd_wrapper import LMDCodec


class CodecFactory:
    """
    Factory pour creer des instances de codecs.

    Supporte la creation de codecs FFmpeg (H.264, H.265, VP9, AV1)
    et du codec LMD-PPV.
    """

    # Registre des codecs disponibles
    _registry: Dict[str, Type[CodecBase]] = {
        'h264': H264Codec,
        'h265': H265Codec,
        'hevc': H265Codec,  # Alias
        'vp9': VP9Codec,
        'av1': AV1Codec,
        'lmd': LMDCodec,
        'lmd-ppv': LMDCodec,
    }

    # Configurations par defaut
    _default_configs: Dict[str, Dict] = {
        'h264': {'preset': 'medium', 'profile': 'high'},
        'h265': {'preset': 'medium'},
        'vp9': {'cpu_used': 2},
        'av1': {'cpu_used': 4},
        'lmd': {'block_size': 16, 'n_colors': 256, 'use_turbo': True},
    }

    @classmethod
    def create(
        cls,
        codec_name: str,
        **kwargs
    ) -> CodecBase:
        """
        Cree une instance de codec.

        Args:
            codec_name: Nom du codec ('h264', 'h265', 'vp9', 'av1', 'lmd')
            **kwargs: Parametres de configuration du codec

        Returns:
            Instance de CodecBase

        Raises:
            ValueError: Si le codec n'est pas supporte
        """
        name_lower = codec_name.lower()

        if name_lower not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Codec '{codec_name}' non supporte. "
                f"Codecs disponibles: {available}"
            )

        # Fusionner avec la configuration par defaut
        default_config = cls._default_configs.get(name_lower, {}).copy()
        default_config.update(kwargs)

        codec_class = cls._registry[name_lower]
        return codec_class(**default_config)

    @classmethod
    def create_all(
        cls,
        include_lmd: bool = True,
        **kwargs
    ) -> Dict[str, CodecBase]:
        """
        Cree une instance de chaque codec disponible.

        Args:
            include_lmd: Inclure le codec LMD-PPV
            **kwargs: Parametres passes a tous les codecs

        Returns:
            Dictionnaire nom -> codec
        """
        codecs = {}

        for name in ['h264', 'h265', 'vp9', 'av1']:
            try:
                codec = cls.create(name, **kwargs)
                if codec.is_available():
                    codecs[name] = codec
            except Exception:
                pass

        if include_lmd:
            try:
                lmd = cls.create('lmd', **kwargs)
                if lmd.is_available():
                    codecs['lmd'] = lmd
            except Exception:
                pass

        return codecs

    @classmethod
    def create_ffmpeg_codecs(cls, **kwargs) -> Dict[str, FFmpegCodec]:
        """
        Cree uniquement les codecs FFmpeg.

        Args:
            **kwargs: Parametres de configuration

        Returns:
            Dictionnaire nom -> FFmpegCodec
        """
        codecs = {}

        for name in ['h264', 'h265', 'vp9', 'av1']:
            try:
                codec = cls.create(name, **kwargs)
                if codec.is_available():
                    codecs[name] = codec
            except Exception:
                pass

        return codecs

    @classmethod
    def register(cls, name: str, codec_class: Type[CodecBase]) -> None:
        """
        Enregistre un nouveau codec.

        Args:
            name: Nom du codec
            codec_class: Classe du codec
        """
        cls._registry[name.lower()] = codec_class

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Liste les codecs disponibles sur le systeme.

        Returns:
            Liste des noms de codecs disponibles
        """
        available = []

        for name in set(cls._registry.keys()):
            try:
                codec = cls.create(name)
                if codec.is_available():
                    available.append(name)
            except Exception:
                pass

        return sorted(set(available))

    @classmethod
    def list_registered(cls) -> List[str]:
        """
        Liste tous les codecs enregistres.

        Returns:
            Liste des noms de codecs enregistres
        """
        return sorted(set(cls._registry.keys()))

    @classmethod
    def get_codec_info(cls, codec_name: str) -> Dict:
        """
        Retourne les informations sur un codec.

        Args:
            codec_name: Nom du codec

        Returns:
            Dictionnaire d'informations
        """
        name_lower = codec_name.lower()

        if name_lower not in cls._registry:
            return {'error': f"Codec '{codec_name}' non supporte"}

        try:
            codec = cls.create(name_lower)
            return {
                'name': codec.name,
                'class': codec.__class__.__name__,
                'available': codec.is_available(),
                'qualities': codec.get_supported_qualities(),
                'extension': codec.get_output_extension(),
                'default_config': cls._default_configs.get(name_lower, {})
            }
        except Exception as e:
            return {'error': str(e)}
