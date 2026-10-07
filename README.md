# precov

A Python reproduction of the pipeline described in

> Li, S., Geringer-Sameth, A., & Golovich, N. (2026),
> *An Automated Probabilistic Asteroid Prediscovery Pipeline*,
> The Astronomical Journal, 172:91, doi:[10.3847/1538-3881/ae7c73](https://doi.org/10.3847/1538-3881/ae7c73)

This package was built as a paired read-and-implement exercise: every module
maps onto a specific section, equation, or algorithm of the paper, and the
docstrings say so explicitly. Where we found the source material ambiguous or
internally inconsistent, we re-derived the relevant result from first
principles (symbolically, via SymPy, where practical) rather than guessing,
and the docstrings explain the reasoning. See **Known departures from the
source material**, below, before trusting any of this for actual research use.

## What this is (and isn't)

This **is**: a faithful, tested implementation of the paper's statistical and
geometric machinery -- the Huber-robust orbit fit, the boundary iso-surface
uncertainty sampling, the convex-hull search-region construction, the
12-integration covariance shortcut, and the Rayleigh-point-process
likelihood-ratio linking test, all validated against independent Monte Carlo
and symbolic ground truth rather than only against each other.

This **is not**: a drop-in replacement for the paper's actual pipeline. The
paper propagates orbits with ASSIST (IAS15 + JPL DE441), capturing full
planetary perturbations, which matters enormously at the multi-year
look-backs this method targets. This package ships a `Propagator` interface
with two implementations:

- `TwoBodyKeplerPropagator` -- pure Sun-only two-body dynamics, no external
  data files. This is what every test in this repository runs against, and
  what the bundled demo uses. **Not adequate for real precovery astrometry.**
- `ASSISTPropagator` -- a real wrapper around `rebound`/`assist`. It requires
  locally-supplied JPL DE441 + asteroid-perturber kernel files (typically
  ~1 GB, obtained separately -- see the `assist` package's own documentation)
  that we could not fetch in the sandbox this was built in, so
  `ASSISTPropagator.geometry` is a documented integration point rather than a
  tested implementation. Wire it up following the control-flow pattern in
  `TwoBodyKeplerPropagator.geometry` once you have the kernel files.

Because both propagators implement the same interface, swapping one for the
other changes nothing else in the pipeline.

## Installation

```bash
pip install -e .[detection,dev]
# add [dynamics] too if you have rebound/assist + local ephemeris kernels
```

## Architecture

```
precov/
├── constants.py          Physical constants + paper defaults (Sec. 3.*)
├── data_models.py         Shared dataclasses (Observation, OrbitFitResult, ...)
├── orbits/
│   ├── elements.py         Two-body Keplerian mechanics (Kepler's equation, state vectors)
│   ├── propagator.py       f(theta, t) -> geometry; TwoBodyKeplerPropagator, ASSISTPropagator
│   ├── fitting.py           Huber-robust orbit fit (Sec. 3.1, Algorithm 1)
│   └── sampling.py           Boundary iso-surface sampling (Sec. 3.2, Algorithm 2)
├── images/
│   ├── photometry.py        IAU H-G apparent magnitude model
│   ├── footprint.py          Viability gating + convex-hull search regions (Sec. 3.2, Algorithm 3)
│   └── detection.py           Low-threshold source extraction (Sec. 3.3, Algorithm 4)
├── linking/
│   ├── covariance.py         12-integration covariance shortcut (Appendix A, Algorithm 5)
│   ├── likelihood.py          Mahalanobis matching + single-image log-likelihood ratio (Eq. 5-8)
│   ├── null_distribution.py   Null-hypothesis moments + significance lookup (Appendix B.1)
│   ├── linker.py               Candidate refit + cross-image linking (Algorithm 6)
│   └── consensus.py             Mutual self-consistency check (Sec. 3.6)
├── survey/
│   ├── base.py                Survey-agnostic image-provider interface
│   └── synthetic.py            Synthetic image generator (testing/demo only)
└── pipeline.py                Top-level orchestrator: PrediscoveryPipeline.run(...)
```

Run the tests to see the whole thing validated stage by stage:

```bash
pytest tests/ -v
```

`tests/test_pipeline_synthetic.py` is the end-to-end check, structurally
mirroring the paper's own Sec. 4 validation: an injected orbit is recovered
across several synthetic archival images at >100-sigma significance and the
correct sub-pixel position (comparable to the paper's own reported >35-sigma
and >45-sigma real recoveries), while a control run with no injected source
(background clutter only) confirms nothing, matching the paper's null tests
on 2022 DB4 / 2022 ED1 (Sec. 4.4, Fig. 5).

## Known departures from the source material

We had two independent renderings of this paper to work from -- the
published PDF and a derived prose report -- and they disagreed with each
other, and in one case the PDF itself was internally inconsistent, in ways
that mattered for implementation. Rather than picking one transcription and
hoping, we re-derived the disputed results directly from the more primitive
equations both documents agreed on, and pinned the re-derivation down with a
test.

1. **Huber loss (Sec. 3.1).** The typeset paper equation for ρ(z,c) is
   dimensionally inconsistent as rendered. We use the classical Huber penalty
   of the standardized residual (`orbits/fitting.py`), verified via the
   pseudo-residual identity `sign(z)*sqrt(rho(z,c)))^2 == rho(z,c)` that lets
   an ordinary least-squares solver perform the robust M-estimation.

2. **Exemplar covariance caching (Appendix A vs. a derived report's
   Algorithm 6 pseudocode).** One transcription computes the shared
   image-pair covariance from the plain postdiscovery fit for every image
   pair; the paper's actual Appendix A computes it from an *exemplar
   candidate's post-refit* orbit, cached per origin image. We followed the
   primary source (`linking/covariance.py::ExemplarCovarianceCache`).

3. **The single-image log-likelihood ratio (Eq. 8/B4/B5) -- the important
   one.** A derived report rendered this as a three-term expression with two
   *different* exponentials. We initially implemented that version; a unit
   test asserting "a distant match in a crowded field should carry
   non-positive evidence" caught it producing nonsensical large-positive
   values. We then derived `f1(r)/f0(r)` symbolically (SymPy) directly from
   the primitive Poisson/Rayleigh definitions in Appendix B, which collapses
   -- exactly, the `exp(-pi*rho_tilde*r^2)` factors cancel -- to a clean
   **two-term, single-exponential** closed form:

   ```
   lambda(r) = log[ (1-alpha) + alpha * (1 + 1/(2*pi*rho_tilde)) * exp(-r^2/2) ]
   ```

   This matches the paper's own printed Eq. (B5) once read as two terms, not
   three. `linking/likelihood.py` and `linking/null_distribution.py` document
   the full derivation and cross-check it symbolically at test time
   (`tests/test_likelihood.py::test_loglik_ratio_matches_symbolic_derivation_from_f0_f1`).

The lesson we'd draw for anyone reproducing a paper from a secondary
transcription: treat *both* documents as untrusted where they touch the
result you're about to implement, and re-derive from whatever primitive
equations they agree on before writing a line of code. A test that encodes
"this quantity should be bounded / should be negative in this regime" is
what actually caught the error here -- self-consistency between the two
transcriptions was not enough, since a wrong three-term reconstruction and a
matching wrong change-of-variables can both be internally consistent with
each other while being wrong relative to the underlying model.

## What's deliberately out of scope

The paper's Sec. 5.1 future-work items -- a signal-matched trailing-source
filter, digital tracking / synthetic co-addition below the single-frame
detection limit, and real ZTF/IRSA archive ingestion at CCD-quadrant
granularity -- are not implemented. The `SurveyProvider` interface is
designed so that a real archive backend (or a matched-filter detector) can be
dropped in without touching the fitting or linking layers.

### ZTF provider

The package now includes `precov.survey.ZTFDifferenceImageProvider` for local
ZTF difference-image FITS files or a CSV manifest of downloaded/cached files.
It extracts the timestamp, WCS, image shape, 3-sigma limiting magnitude, PSF
FWHM, and observer location from the FITS header when possible, and falls
back to manifest values when needed.

Typical usage is:

1. Download the ZTF difference-image quadrants you want to search from IRSA.
2. Save them locally and write a CSV manifest with `image_id` plus either
   `path` or `url`.
3. Include `time`, `limiting_mag_3sigma`, and `fwhm_px` in the manifest when
   those values are not present in the FITS headers.
4. Pass the resulting provider to `PrediscoveryPipeline`.

The other inputs you still need are the postdiscovery MPC astrometry and the
asteroid's `H`/`G` values. The survey provider supplies the image-side data.

For a full step-by-step walkthrough, see [docs/ztf_recovery_guide.md](docs/ztf_recovery_guide.md).
# prevcov-git
