"""
precov: A probabilistic asteroid prediscovery pipeline.

Reproduction of the method described in Li, Geringer-Sameth & Golovich (2026),
"An Automated Probabilistic Asteroid Prediscovery Pipeline", AJ 172:91,
doi:10.3847/1538-3881/ae7c73.

The package is organized around the four conceptual stages of the paper:

    precov.orbits    -- robust orbit fitting (Sec. 3.1) and boundary
                         iso-surface uncertainty sampling (Sec. 3.2)
    precov.images    -- photometric/geometric viability gating, convex-hull
                         search regions, and low-threshold source extraction
                         (Sec. 3.2-3.3)
    precov.linking   -- fast linearized covariance mapping, Mahalanobis
                         matching, and the standardized log-likelihood-ratio
                         test (Sec. 3.4-3.6, Appendices A & B)
    precov.survey    -- survey-agnostic image/catalog provider interface

`precov.pipeline.PrediscoveryPipeline` wires these together end to end.

Two derivations in this codebase deliberately depart from the literal
OCR'd equations in the source material, in favor of independently re-derived
and numerically verified forms. See the docstrings of
`precov.orbits.fitting.huber_rho` and `precov.linking.null_distribution`
for the reasoning.
"""

from .version import __version__

__all__ = ["__version__"]
