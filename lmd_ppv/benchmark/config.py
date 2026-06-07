"""
config.py - Configuration globale du benchmark
==============================================

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import json


@dataclass
class DatasetConfig:
    """Configuration d'un dataset."""
    name: str
    source: str  # 'xiph', 'cdvl', 'vimeo'
    base_url: str
    videos: List[str] = field(default_factory=list)
    resolutions: List[str] = field(default_factory=list)  # e.g., ['cif', '720p', '1080p']


@dataclass
class CodecConfig:
    """Configuration d'un codec."""
    name: str
    ffmpeg_encoder: str  # e.g., 'libx264'
    crf_range: List[int] = field(default_factory=list)  # e.g., [18, 23, 28]
    preset: str = 'medium'
    extra_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration complete du benchmark."""

    # Repertoires
    output_dir: Path = field(default_factory=lambda: Path('./benchmark_results'))
    datasets_dir: Path = field(default_factory=lambda: Path('./datasets'))
    cache_dir: Path = field(default_factory=lambda: Path('./cache'))

    # Split train/test
    train_ratio: float = 0.7
    random_seed: int = 42

    # Codecs a tester
    codecs: Dict[str, CodecConfig] = field(default_factory=dict)

    # Parallelisation
    n_workers: int = 4
    max_videos: Optional[int] = None  # Limite pour tests rapides

    # Metriques
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_vmaf: bool = False  # Necessite installation separee

    def __post_init__(self):
        """Initialise les configurations par defaut."""
        if not self.codecs:
            self.codecs = self._default_codecs()

        # Creer les repertoires
        for dir_path in [self.output_dir, self.datasets_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_codecs() -> Dict[str, CodecConfig]:
        """Codecs par defaut."""
        return {
            'h264': CodecConfig(
                name='H.264/AVC',
                ffmpeg_encoder='libx264',
                crf_range=[18, 23, 28],
                preset='medium',
                extra_params={'tune': 'film'}
            ),
            'h265': CodecConfig(
                name='H.265/HEVC',
                ffmpeg_encoder='libx265',
                crf_range=[18, 23, 28],
                preset='medium',
                extra_params={'x265-params': 'log-level=error'}
            ),
            'vp9': CodecConfig(
                name='VP9',
                ffmpeg_encoder='libvpx-vp9',
                crf_range=[20, 30, 40],
                preset='good',
                extra_params={'b:v': '0', 'cpu-used': '2'}
            ),
            'av1': CodecConfig(
                name='AV1',
                ffmpeg_encoder='libaom-av1',
                crf_range=[25, 35, 45],
                preset='4',  # cpu-used
                extra_params={'strict': 'experimental', 'cpu-used': '4'}
            ),
        }

    @classmethod
    def default_datasets(cls) -> List[DatasetConfig]:
        """Datasets par defaut."""
        return [
            DatasetConfig(
                name='xiph',
                source='xiph',
                base_url='https://media.xiph.org/video/derf/',
                resolutions=['cif', '720p', '1080p'],
                videos=[
                    # CIF (352x288)
                    'akiyo_cif.y4m',
                    'bowing_cif.y4m',
                    'bridge-close_cif.y4m',
                    'bridge-far_cif.y4m',
                    'bus_cif.y4m',
                    'coastguard_cif.y4m',
                    'container_cif.y4m',
                    'crew_cif.y4m',
                    'flower_cif.y4m',
                    'foreman_cif.y4m',
                    'hall_cif.y4m',
                    'highway_cif.y4m',
                    'mobile_cif.y4m',
                    'mother-daughter_cif.y4m',
                    'news_cif.y4m',
                    'paris_cif.y4m',
                    'silent_cif.y4m',
                    'stefan_cif.y4m',
                    'tempete_cif.y4m',
                    'waterfall_cif.y4m',
                    # 720p
                    'crowd_run_720p50.y4m',
                    'ducks_take_off_720p50.y4m',
                    'in_to_tree_720p50.y4m',
                    'old_town_cross_720p50.y4m',
                    'park_joy_720p50.y4m',
                    'pedestrian_area_720p25.y4m',
                    # 1080p
                    'blue_sky_1080p25.y4m',
                    'crowd_run_1080p50.y4m',
                    'ducks_take_off_1080p50.y4m',
                    'in_to_tree_1080p50.y4m',
                    'old_town_cross_1080p50.y4m',
                    'park_joy_1080p50.y4m',
                    'pedestrian_area_1080p25.y4m',
                    'red_kayak_1080p.y4m',
                    'riverbed_1080p25.y4m',
                    'rush_hour_1080p25.y4m',
                    'station2_1080p25.y4m',
                    'sunflower_1080p25.y4m',
                    'tractor_1080p25.y4m',
                ]
            ),
            DatasetConfig(
                name='cdvl',
                source='cdvl',
                base_url='https://cdvl.org/downloads/',
                resolutions=['720p', '1080p'],
                videos=[]  # Will be populated during download scan
            ),
            DatasetConfig(
                name='vimeo90k',
                source='vimeo',
                base_url='',  # Requires manual download
                resolutions=['448p'],
                videos=[]  # Subset selection during processing
            ),
        ]

    def save(self, path: Path) -> None:
        """Sauvegarde la configuration en JSON."""
        config_dict = {
            'output_dir': str(self.output_dir),
            'datasets_dir': str(self.datasets_dir),
            'cache_dir': str(self.cache_dir),
            'train_ratio': self.train_ratio,
            'random_seed': self.random_seed,
            'n_workers': self.n_workers,
            'max_videos': self.max_videos,
            'compute_psnr': self.compute_psnr,
            'compute_ssim': self.compute_ssim,
            'compute_vmaf': self.compute_vmaf,
            'codecs': {
                name: {
                    'name': c.name,
                    'ffmpeg_encoder': c.ffmpeg_encoder,
                    'crf_range': c.crf_range,
                    'preset': c.preset,
                    'extra_params': c.extra_params
                }
                for name, c in self.codecs.items()
            }
        }
        path.write_text(json.dumps(config_dict, indent=2))

    @classmethod
    def load(cls, path: Path) -> 'BenchmarkConfig':
        """Charge la configuration depuis un JSON."""
        data = json.loads(path.read_text())

        codecs = {}
        for name, c in data.get('codecs', {}).items():
            codecs[name] = CodecConfig(
                name=c['name'],
                ffmpeg_encoder=c['ffmpeg_encoder'],
                crf_range=c.get('crf_range', []),
                preset=c.get('preset', 'medium'),
                extra_params=c.get('extra_params', {})
            )

        return cls(
            output_dir=Path(data.get('output_dir', './benchmark_results')),
            datasets_dir=Path(data.get('datasets_dir', './datasets')),
            cache_dir=Path(data.get('cache_dir', './cache')),
            train_ratio=data.get('train_ratio', 0.7),
            random_seed=data.get('random_seed', 42),
            n_workers=data.get('n_workers', 4),
            max_videos=data.get('max_videos'),
            compute_psnr=data.get('compute_psnr', True),
            compute_ssim=data.get('compute_ssim', True),
            compute_vmaf=data.get('compute_vmaf', False),
            codecs=codecs
        )


# Configuration par defaut
DEFAULT_CONFIG = BenchmarkConfig()
