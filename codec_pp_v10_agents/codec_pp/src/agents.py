"""
PPV Agent System — Système multi-agent pour l'optimisation itérative du codec.

Architecture :
═══════════════

                    ┌──────────────────┐
                    │   ORCHESTRATOR   │
                    │  (cycle maître)  │
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │  PROPOSER   │  │  EVALUATOR  │  │  CATALOGER  │
    │  (Axe 8)    │  │  (Axe 9)   │  │  (Axe 10)   │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                 │                 │
    ┌──────▼──────────────────▼─────────────────▼──────┐
    │                 SPECIALIST AGENTS                 │
    ├──────────┬──────────┬──────────┬────────────────┤
    │ Encoder  │ Decoder  │ Precoder │ Compressor     │
    │ (Axe 1)  │ (Axe 1) │ (Axe 2)  │ (Axe 4)        │
    ├──────────┼──────────┼──────────┼────────────────┤
    │ LowLevel │ Trainer  │ Builder  │ Bench          │
    │ (Axe 5)  │ (Axe 6) │ (Axe 7)  │ (Axe 5+9)     │
    └──────────┴──────────┴──────────┴────────────────┘

Protocole de communication :
  - Chaque agent expose des ACTIONS (fonctions appelables)
  - Chaque agent émet des EVENTS (résultats, métriques, erreurs)
  - L'orchestrateur maintient un STATE global (version courante,
    métriques de référence, règles actives, historique)
  - Les messages sont des dicts JSON sérialisables

Cycle d'itération (boucle principale de l'orchestrateur) :
  1. PROPOSER.suggest_rule()      → candidate_rule
  2. ENCODER.apply_rule(rule)     → modified_codec
  3. BENCH.run_corpus(codec)      → raw_metrics
  4. EVALUATOR.analyze(metrics)   → delta_report
  5. ORCHESTRATOR.decide(report)  → accept / reject
  6. CATALOGER.record(version)    → catalog entry
  7. → retour à 1 si non convergé
"""

import json
import time
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
#  TYPES DE BASE
# ═══════════════════════════════════════════════════════════════

class RuleStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class Rule:
    """Une règle d'optimisation candidate."""
    name: str
    axe: int               # 1-10
    parameter: str         # ex: "prediction_mode"
    value: Any             # ex: "paeth"
    description: str
    status: RuleStatus = RuleStatus.PROPOSED
    gain_bpp: float = 0.0
    gain_ratio: float = 0.0
    confidence: float = 0.0
    tested_on: int = 0     # nombre de vidéos testées

    def to_dict(self):
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass
class VersionEntry:
    """Entrée dans le catalogue des versions."""
    version_id: str
    timestamp: str
    parent_id: Optional[str]
    rules_active: List[str]
    metrics: Dict[str, float]
    delta_vs_parent: Dict[str, float]
    delta_vs_h264: Dict[str, float]
    config_hash: str
    notes: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentMessage:
    """Message échangé entre agents."""
    sender: str
    action: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemState:
    """État global du système."""
    current_version: str = "v9.0"
    iteration: int = 0
    converged: bool = False
    convergence_counter: int = 0  # itérations sans gain > seuil
    convergence_threshold: float = 0.001  # 0.1% gain minimum
    max_convergence_stalls: int = 5
    rules_active: List[Rule] = field(default_factory=list)
    rules_history: List[Rule] = field(default_factory=list)
    reference_metrics: Dict[str, float] = field(default_factory=dict)
    h264_reference: Dict[str, float] = field(default_factory=dict)
    catalog: List[VersionEntry] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  AGENTS
# ═══════════════════════════════════════════════════════════════

class BaseAgent:
    """Agent de base avec logging et communication."""

    def __init__(self, name: str, state: SystemState):
        self.name = name
        self.state = state
        self.log: List[str] = []

    def emit(self, action: str, payload: Dict) -> AgentMessage:
        msg = AgentMessage(sender=self.name, action=action, payload=payload)
        self.log.append(f"[{self.name}] {action}: {json.dumps(payload, default=str)[:120]}")
        return msg

    def info(self, text: str):
        self.log.append(f"[{self.name}] {text}")


class ProposerAgent(BaseAgent):
    """
    Agent Proposeur (Axe 8) — Explore l'espace des optimisations.

    Maintient une liste de règles candidates ordonnées par gain estimé.
    Utilise les métriques bloc (Axe 9) pour prioriser.
    """

    def __init__(self, state: SystemState):
        super().__init__("PROPOSER", state)
        self.rule_bank = self._init_rule_bank()
        self.rules_tested = set()

    def _init_rule_bank(self) -> List[Rule]:
        """Initialise la banque de règles candidates."""
        return [
            # Axe 2 — Précodage
            Rule("pred_paeth", 2, "prediction_mode", "paeth",
                 "Prédiction Paeth (PNG) au lieu de Left/Top/DC"),
            Rule("pred_med", 2, "prediction_mode", "med",
                 "Prédiction MED (JPEG-LS) : median edge-detecting"),
            Rule("pred_gradient", 2, "prediction_mode", "gradient",
                 "Prédiction gradient : left + top - topleft"),
            Rule("delta_temporal", 2, "temporal_preprocess", "delta",
                 "Delta temporel : encoder seq[t] - seq[t-1] mod 256"),
            Rule("xor_temporal", 2, "temporal_preprocess", "xor",
                 "XOR temporel : seq[t] XOR seq[t-1]"),

            # Axe 4 — Compression MDL
            Rule("rep_r5_runlength", 4, "representation", "r5",
                 "R5 : Run-length + palette pour segments longs"),
            Rule("rep_r6_delta_color", 4, "representation", "r6",
                 "R6 : Delta couleur pour dégradés temporels"),
            Rule("arithmetic_coding", 4, "entropy_coder", "arithmetic",
                 "Codage arithmétique au lieu de Huffman"),
            Rule("code_methode_huffman", 4, "code_methode_compression", True,
                 "Huffman sur les Code Méthode (métadonnées)"),
            Rule("per_pixel_rep", 4, "rep_granularity", "pixel",
                 "Choix de R par pixel au lieu de par bloc"),

            # Axe 2 — Palette
            Rule("palette_threshold_32", 2, "palette_threshold", 32,
                 "Palette si m ≤ 32 (au lieu de 64)"),
            Rule("palette_threshold_16", 2, "palette_threshold", 16,
                 "Palette si m ≤ 16 (strict)"),

            # Axe 2 — Fenêtres
            Rule("window_max_64", 2, "max_window", 64,
                 "Fenêtre adaptative max 64 frames"),
            Rule("window_max_128", 2, "max_window", 128,
                 "Fenêtre adaptative max 128 frames"),
            Rule("window_ncd_split", 2, "window_criterion", "ncd_gradient",
                 "Couper la fenêtre quand le gradient de NCD > seuil"),

            # Axe 4 — Block size
            Rule("block_4x4", 4, "block_size", 4,
                 "Blocs 4×4 au lieu de 8×8"),
            Rule("block_16x16", 4, "block_size", 16,
                 "Blocs 16×16 au lieu de 8×8"),
            Rule("block_adaptive", 4, "block_size", "adaptive",
                 "Taille de bloc adaptative selon le contenu"),

            # Axe 5 — Bas niveau
            Rule("bitpack_uint64", 5, "bool_vector_type", "uint64",
                 "Vecteur booléen sur uint64_t avec POPCNT"),
            Rule("simd_palette", 5, "palette_lookup", "simd",
                 "Lookup palette vectorisé AVX2"),
        ]

    def suggest_rule(self) -> Optional[Rule]:
        """Suggère la prochaine règle à tester."""
        untested = [r for r in self.rule_bank if r.name not in self.rules_tested]

        if not untested:
            self.info("Toutes les règles ont été testées")
            return None

        # Prioriser par axe (2 et 4 d'abord = ratio pur)
        priority = {2: 0, 4: 1, 5: 2, 1: 3}
        untested.sort(key=lambda r: (priority.get(r.axe, 9), r.name))

        rule = untested[0]
        rule.status = RuleStatus.TESTING
        self.rules_tested.add(rule.name)

        self.emit("suggest_rule", {"rule": rule.name, "axe": rule.axe,
                                    "param": rule.parameter, "value": str(rule.value)})
        return rule

    def receive_evaluation(self, rule: Rule, delta: Dict):
        """Reçoit le résultat de l'évaluation et met à jour la règle."""
        rule.gain_bpp = delta.get('avg_bpp_delta', 0)
        rule.gain_ratio = delta.get('avg_ratio_delta', 0)
        rule.confidence = delta.get('confidence', 0)
        rule.tested_on = delta.get('n_videos', 0)

        if rule.gain_bpp > 0:
            rule.status = RuleStatus.ACCEPTED
            self.info(f"Règle {rule.name} ACCEPTÉE : gain bpp = {rule.gain_bpp:+.3f}")
        else:
            rule.status = RuleStatus.REJECTED
            self.info(f"Règle {rule.name} REJETÉE : gain bpp = {rule.gain_bpp:+.3f}")


class EvaluatorAgent(BaseAgent):
    """
    Agent Évaluateur (Axe 9) — Calcule les métriques et analyse les deltas.

    Reçoit les résultats bruts du benchmark et produit un rapport structuré.
    """

    def __init__(self, state: SystemState):
        super().__init__("EVALUATOR", state)

    def analyze(
        self,
        metrics_new: Dict[str, Dict],
        metrics_ref: Dict[str, Dict],
        metrics_h264: Dict[str, Dict],
    ) -> Dict:
        """
        Analyse les métriques et produit un rapport de delta.

        Args:
            metrics_new: {video_name: {bpp, ratio, psnr, ...}}
            metrics_ref: métriques de la version de référence
            metrics_h264: métriques H.264 lossless

        Returns:
            rapport avec deltas, confiance, recommandation
        """
        deltas = {}
        bpp_gains = []
        ratio_gains = []
        regressions = []

        for video, m_new in metrics_new.items():
            m_ref = metrics_ref.get(video, {})
            m_h264 = metrics_h264.get(video, {})

            delta_bpp = m_ref.get('bpp', 8) - m_new.get('bpp', 8)
            delta_ratio = m_new.get('ratio', 1) / max(m_ref.get('ratio', 1), 0.01) - 1

            bpp_gains.append(delta_bpp)
            ratio_gains.append(delta_ratio)

            if delta_ratio < -0.01:  # régression > 1%
                regressions.append(video)

            deltas[video] = {
                'bpp_delta': delta_bpp,
                'ratio_delta': delta_ratio,
                'vs_h264_ratio': m_new.get('ratio', 1) / max(m_h264.get('ratio', 1), 0.01),
            }

        import numpy as np
        avg_bpp = float(np.mean(bpp_gains)) if bpp_gains else 0
        avg_ratio = float(np.mean(ratio_gains)) if ratio_gains else 0
        std_ratio = float(np.std(ratio_gains)) if ratio_gains else 0

        report = {
            'avg_bpp_delta': avg_bpp,
            'avg_ratio_delta': avg_ratio,
            'std_ratio_delta': std_ratio,
            'confidence': max(0, 1 - std_ratio / max(abs(avg_ratio), 0.001)),
            'n_videos': len(metrics_new),
            'n_regressions': len(regressions),
            'regressions': regressions,
            'deltas': deltas,
            'recommendation': 'accept' if avg_ratio > 0.001 and len(regressions) == 0 else 'reject',
        }

        self.emit("analysis_complete", {
            "avg_bpp": f"{avg_bpp:+.4f}",
            "avg_ratio": f"{avg_ratio:+.2%}",
            "recommendation": report['recommendation'],
        })

        return report


class CatalogerAgent(BaseAgent):
    """
    Agent Catalogueur (Axe 10) — Gère le catalogue des versions.
    """

    def __init__(self, state: SystemState, catalog_path: str = "ppv_catalog.json"):
        super().__init__("CATALOGER", state)
        self.catalog_path = catalog_path

    def record_version(
        self,
        rules_active: List[Rule],
        metrics: Dict[str, float],
        delta_vs_parent: Dict[str, float],
        delta_vs_h264: Dict[str, float],
        notes: str = "",
    ) -> VersionEntry:
        """Enregistre une nouvelle version dans le catalogue."""
        # Générer un ID de version
        config_str = json.dumps([r.name for r in rules_active], sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        parent_id = self.state.catalog[-1].version_id if self.state.catalog else None

        entry = VersionEntry(
            version_id=f"v{9 + self.state.iteration}.{len(self.state.catalog)}",
            timestamp=datetime.now().isoformat(),
            parent_id=parent_id,
            rules_active=[r.name for r in rules_active],
            metrics=metrics,
            delta_vs_parent=delta_vs_parent,
            delta_vs_h264=delta_vs_h264,
            config_hash=config_hash,
            notes=notes,
        )

        self.state.catalog.append(entry)
        self._save_catalog()

        self.emit("version_recorded", {
            "version_id": entry.version_id,
            "config_hash": config_hash,
            "n_rules": len(rules_active),
        })

        return entry

    def _save_catalog(self):
        """Sauvegarde le catalogue en JSON."""
        data = [e.to_dict() for e in self.state.catalog]
        with open(self.catalog_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_catalog(self):
        """Charge le catalogue depuis le fichier."""
        if os.path.exists(self.catalog_path):
            with open(self.catalog_path) as f:
                data = json.load(f)
            self.state.catalog = [VersionEntry(**d) for d in data]

    def get_progression(self) -> List[Dict]:
        """Retourne la progression des métriques au fil des versions."""
        return [
            {
                'version': e.version_id,
                'ratio_mean': e.metrics.get('ratio_mean', 0),
                'bpp_mean': e.metrics.get('bpp_mean', 0),
                'n_rules': len(e.rules_active),
            }
            for e in self.state.catalog
        ]


class BenchAgent(BaseAgent):
    """
    Agent Benchmark (Axe 5+9) — Exécute l'encodage sur le corpus.
    """

    def __init__(self, state: SystemState):
        super().__init__("BENCH", state)

    def run_corpus(
        self,
        codec_config: Dict,
        corpus: List[Dict],
    ) -> Dict[str, Dict]:
        """
        Encode chaque vidéo du corpus et retourne les métriques.

        Args:
            codec_config: configuration de l'encodeur
            corpus: liste de {name, data, fps, ...}

        Returns:
            {video_name: {bpp, ratio, psnr, encode_ms, ...}}
        """
        from codec_pp.src.encoder import PPVEncoder
        from codec_pp.src.decoder import PPVDecoder
        import numpy as np
        import tempfile

        results = {}
        ppv_path = tempfile.mktemp(suffix='.ppv')

        enc = PPVEncoder(
            gop_size=codec_config.get('gop_size', 32),
            block_size=codec_config.get('block_size', 8),
            use_huffman=codec_config.get('use_huffman', True),
            verbose=False,
        )
        dec = PPVDecoder(verbose=False)

        for video in corpus:
            name = video['name']
            M = video['data']

            t0 = time.time()
            stats = enc.encode(M=M, output_path=ppv_path, color_bits=8)
            enc_ms = (time.time() - t0) * 1000

            t0 = time.time()
            M_dec, meta = dec.decode(ppv_path)
            dec_ms = (time.time() - t0) * 1000

            lossless = np.array_equal(M, M_dec)

            results[name] = {
                'bpp': stats['compressed_size_bytes'] * 8 / M.size,
                'ratio': stats['compression_ratio'],
                'savings': stats['savings_percent'],
                'encode_ms': enc_ms,
                'decode_ms': dec_ms,
                'lossless': lossless,
                'mono_count': stats['mono_count'],
                'rep_counts': stats['rep_counts'],
            }

            self.info(f"{name}: ratio={stats['compression_ratio']:.2f}x "
                      f"bpp={results[name]['bpp']:.2f}")

        if os.path.exists(ppv_path):
            os.remove(ppv_path)

        return results


# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATEUR
# ═══════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Agent orchestrateur — coordonne le cycle d'amélioration itérative.

    Boucle :
      1. Proposer une règle
      2. Appliquer la règle (modifier la config de l'encodeur)
      3. Encoder le corpus
      4. Évaluer les métriques
      5. Décider (accepter/rejeter)
      6. Cataloguer
      7. Vérifier convergence
    """

    def __init__(self, corpus: List[Dict], h264_reference: Dict[str, Dict]):
        self.state = SystemState()
        self.state.h264_reference = h264_reference

        self.proposer = ProposerAgent(self.state)
        self.evaluator = EvaluatorAgent(self.state)
        self.cataloger = CatalogerAgent(self.state)
        self.bench = BenchAgent(self.state)

        self.corpus = corpus
        self.current_config = {
            'gop_size': 32,
            'block_size': 8,
            'use_huffman': True,
        }

        # Exécuter le benchmark de référence
        self.info("Benchmark de référence (v9 baseline)...")
        self.state.reference_metrics = self.bench.run_corpus(
            self.current_config, self.corpus
        )

    def info(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] ORCHESTRATOR: {text}")

    def run_iteration(self) -> Dict:
        """Exécute une itération du cycle d'optimisation."""
        self.state.iteration += 1
        self.info(f"═══ Itération {self.state.iteration} ═══")

        # 1. Proposer
        rule = self.proposer.suggest_rule()
        if rule is None:
            self.state.converged = True
            self.info("Plus de règles à tester — convergence")
            return {'converged': True, 'reason': 'no_more_rules'}

        self.info(f"Règle candidate : {rule.name} ({rule.description})")

        # 2. Appliquer (pour l'instant, simulé via config)
        test_config = dict(self.current_config)
        if rule.parameter == "block_size" and isinstance(rule.value, int):
            test_config['block_size'] = rule.value

        # 3. Encoder le corpus
        self.info(f"Encodage du corpus ({len(self.corpus)} vidéos)...")
        metrics_new = self.bench.run_corpus(test_config, self.corpus)

        # 4. Évaluer
        report = self.evaluator.analyze(
            metrics_new,
            self.state.reference_metrics,
            self.state.h264_reference,
        )

        # 5. Décider
        self.proposer.receive_evaluation(rule, report)

        if report['recommendation'] == 'accept':
            self.info(f"✓ ACCEPTÉ : gain ratio moyen = {report['avg_ratio_delta']:+.2%}")
            self.state.rules_active.append(rule)
            self.current_config = test_config
            self.state.reference_metrics = metrics_new
            self.state.convergence_counter = 0
        else:
            self.info(f"✗ REJETÉ : gain ratio moyen = {report['avg_ratio_delta']:+.2%}"
                      f" (régressions : {report['n_regressions']})")
            self.state.convergence_counter += 1

        self.state.rules_history.append(rule)

        # 6. Cataloguer
        import numpy as np
        avg_metrics = {
            'ratio_mean': float(np.mean([m['ratio'] for m in metrics_new.values()])),
            'bpp_mean': float(np.mean([m['bpp'] for m in metrics_new.values()])),
        }
        entry = self.cataloger.record_version(
            rules_active=self.state.rules_active,
            metrics=avg_metrics,
            delta_vs_parent={'avg_ratio_delta': report['avg_ratio_delta']},
            delta_vs_h264={},
            notes=f"Tested rule: {rule.name} → {rule.status.value}",
        )

        # 7. Convergence
        if self.state.convergence_counter >= self.state.max_convergence_stalls:
            self.state.converged = True
            self.info(f"Convergence atteinte après {self.state.iteration} itérations "
                      f"({self.state.convergence_counter} stalls consécutifs)")

        return {
            'iteration': self.state.iteration,
            'rule': rule.name,
            'status': rule.status.value,
            'gain': report['avg_ratio_delta'],
            'converged': self.state.converged,
            'version': entry.version_id,
        }

    def run_until_convergence(self, max_iterations: int = 50) -> List[Dict]:
        """Exécute le cycle complet jusqu'à convergence."""
        results = []
        for _ in range(max_iterations):
            result = self.run_iteration()
            results.append(result)
            if self.state.converged:
                break

        self.info(f"\n{'═' * 60}")
        self.info(f"TERMINÉ — {self.state.iteration} itérations")
        self.info(f"Règles acceptées : {len(self.state.rules_active)}")
        for r in self.state.rules_active:
            self.info(f"  ✓ {r.name} : {r.description}")
        self.info(f"Règles rejetées : "
                  f"{len([r for r in self.state.rules_history if r.status == RuleStatus.REJECTED])}")
        self.info(f"{'═' * 60}")

        return results
