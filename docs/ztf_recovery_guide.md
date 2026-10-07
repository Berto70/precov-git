# ZTF recovery and analysis guide

This guide shows how to use `precov` to recover ZTF difference images and run
prediscovery analysis on them.

The short version:

1. Gather the post-discovery asteroid astrometry.
2. Obtain ZTF difference-image quadrants for the epochs you want to search.
3. Build a `SurveyProvider` from either local FITS files or IRSA products.
4. Run `PrediscoveryPipeline` on those images.

## What the package needs

For each asteroid, you need two kinds of inputs:

### 1) Post-discovery orbit data

These are the MPC astrometric observations used to fit the initial orbit:

- epoch (`astropy.time.Time`)
- right ascension and declination
- 1-sigma astrometric uncertainties
- optionally the observer location for topocentric observations

In `precov`, these are represented by `precov.data_models.Observation`.

### 2) Survey image data

For each ZTF image / CCD quadrant you want to search, the pipeline needs:

- image timestamp
- FITS WCS
- image shape
- 3-sigma limiting magnitude
- PSF FWHM in pixels
- optional observer location
- the pixel array itself

These are represented by `precov.data_models.SurveyImage` and loaded through a
`SurveyProvider` implementation.

## ZTF workflow options

`precov` supports two ways to work with ZTF difference images:

- **Local FITS mode**: point the provider at a directory of FITS files, or at a
  CSV manifest that lists local files.
- **IRSA-backed mode**: point the provider at a manifest or a list of download
  URLs coming from IRSA/ZTF products. The provider caches the files locally and
  uses the same FITS parsing path as the local workflow.

The current ZTF provider is `precov.survey.ZTFDifferenceImageProvider`.

## Installation

If you are using the same conda environment as the repo, activate it first:

```bash
conda activate prerec
```

Then make sure the package is installed in editable mode:

```bash
pip install -e .[detection,dev]
```

If you want the more realistic orbital propagator used by the paper, also
install the dynamics extra and the required `assist`/`rebound` dependencies:

```bash
pip install -e .[detection,dev,dynamics]
```

## Local FITS workflow

If you already downloaded the ZTF difference images to disk, this is the
simplest path.

### Directory layout

You can store the FITS files anywhere, for example:

```text
ztf_data/
  ztf_2018_001.fits
  ztf_2018_002.fits
  ztf_2018_003.fits
```

### Create the provider

```python
from precov.survey import ZTFDifferenceImageProvider

provider = ZTFDifferenceImageProvider.from_directory("/path/to/ztf_data")
```

The provider reads the FITS header to infer:

- `t` from `MJD-OBS`, `JD`, or `DATE-OBS`
- WCS from the header
- `limiting_mag_3sigma` from `MAGLIM`, `LIMMAG`, or a similar keyword
- `fwhm_px` from `FWHM` or `SEEING`
- observer location from `OBSGEO-X/Y/Z` when present

If some of those values are missing from the FITS headers, you can provide them
in a CSV manifest instead.

### Use a CSV manifest

A manifest is useful when metadata is not stored cleanly in the FITS headers.
A minimal manifest can look like this:

```csv
image_id,path,time,limiting_mag_3sigma,fwhm_px
ztf_001,/data/ztf/ztf_001.fits,2024-01-02T03:04:05,20.6,2.1
ztf_002,/data/ztf/ztf_002.fits,2024-01-02T03:34:05,20.5,2.0
```

Then load it with:

```python
provider = ZTFDifferenceImageProvider.from_manifest("/path/to/ztf_manifest.csv")
```

The manifest can also use `url` or `download_url` instead of `path` if you want
cached downloads.

## IRSA-backed workflow

This package does **not** query IRSA for you yet. Instead, it gives you a clean
way to consume products exported from IRSA.

Typical archive workflow:

1. Search IRSA/ZTF for the sky positions and epochs you care about.
2. Export or copy the difference-image product URLs, or save the result table as
  a CSV manifest.
3. Feed that manifest or URL list into the provider.

### Option A: IRSA manifest

If you have a CSV table from IRSA with columns like `image_id`, `url`,
`time`, `limiting_mag_3sigma`, and `fwhm_px`, load it directly:

```python
from precov.survey import ZTFDifferenceImageProvider

provider = ZTFDifferenceImageProvider.from_irsa_manifest("/path/to/irsa_export.csv")
```

This also works if the manifest is hosted at a URL.

If IRSA gives you a table in another delimiter format, convert it to CSV first.

### Option B: explicit URL list

If you already have the download URLs, you can build the provider directly:

```python
from precov.survey import ZTFDifferenceImageProvider

urls = [
    "https://.../ztf_quadrant_001.fits",
    "https://.../ztf_quadrant_002.fits",
]

provider = ZTFDifferenceImageProvider.from_irsa_url_list(
    urls,
    cache_dir="/path/to/cache",
    limiting_mag_3sigma=20.6,
    fwhm_px=2.1,
)
```

If you already know the image IDs, pass them too:

```python
provider = ZTFDifferenceImageProvider.from_irsa_url_list(
    urls,
    image_ids=["ztf_001", "ztf_002"],
    cache_dir="/path/to/cache",
)
```

## Running the analysis

Once you have a provider, the pipeline runs the same way for local and IRSA
inputs.

### 1) Fit the post-discovery astrometry

Create your `Observation` list from the MPC data.

### 2) Create a propagator

For testing or quick development, use the built-in two-body propagator.
For research-grade results, use the ASSIST backend.

```python
from astropy.time import Time
from precov.orbits.propagator import TwoBodyKeplerPropagator

propagator = TwoBodyKeplerPropagator(epoch=Time("2024-01-01T00:00:00", scale="tdb"))
```

### 3) Run the pipeline

```python
import numpy as np
from precov.pipeline import PrediscoveryPipeline

pipeline = PrediscoveryPipeline(
    propagator=propagator,
    survey=provider,
    H=18.0,
    G=0.15,
)

result = pipeline.run(postdiscovery_obs, theta0_guess=np.array([...]))
```

The main outputs are:

- `result.postdiscovery_fit`: best-fit orbit from the MPC data
- `result.n_search_images`: number of viable survey images searched
- `result.n_candidate_sources`: total number of detected candidate sources
- `result.confirmed`: confirmed prediscovery candidates
- `result.joint_fit`: orbit fit after adding the confirmed prediscovery sources

## Recommended practical workflow

A good way to work is:

1. Start with the synthetic example in `examples/demo_synthetic_recovery.py`.
2. Switch to a small local FITS subset and verify the provider can read the
   headers and images.
3. Move to a manifest-based IRSA workflow.
4. Once the archive ingestion is stable, run the full pipeline on a real
   object.

## What ZTF data you should recover

For the paper's style of analysis, the most useful products are:

- ZTF difference-image quadrants
- the corresponding FITS headers with WCS and observing metadata
- any product table or manifest that maps epochs to image files

You usually do **not** need the full raw focal-plane exposure if you already
have the difference-image quadrants and their headers.

## Notes and caveats

- The archive-facing helper methods in this package are intentionally simple:
  they load manifests or URLs and then reuse the same FITS code path.
- If a header is missing `MAGLIM` / `FWHM` / `DATE-OBS`, supply those values in
  the manifest.
- If you want the most faithful orbital propagation, use the ASSIST backend
  rather than the analytic two-body propagator.
- The package is survey-agnostic, so the same workflow can be adapted to other
  archives once you can provide image metadata and pixel arrays.

## Minimal end-to-end example

```python
import numpy as np
from astropy.time import Time
from precov.data_models import Observation
from precov.orbits.propagator import TwoBodyKeplerPropagator
from precov.pipeline import PrediscoveryPipeline
from precov.survey import ZTFDifferenceImageProvider

# Survey images
provider = ZTFDifferenceImageProvider.from_manifest("/path/to/ztf_manifest.csv")

# Post-discovery astrometry
postdiscovery_obs = [
    Observation(
        t=Time("2024-01-01T00:00:00", scale="utc"),
        ra=1.0,
        dec=0.5,
        sigma_ra=np.radians(1.0 / 3600.0),
        sigma_dec=np.radians(1.0 / 3600.0),
        robust=True,
    ),
]

# Propagator and pipeline
propagator = TwoBodyKeplerPropagator(epoch=Time("2024-01-01T00:00:00", scale="tdb"))
pipeline = PrediscoveryPipeline(propagator=propagator, survey=provider, H=18.0)

# Initial orbit guess
theta0_guess = np.array([1.5, 0.2, 0.1, 0.1, 0.1, 0.1])
result = pipeline.run(postdiscovery_obs, theta0_guess)

print(result.n_search_images)
print(result.n_candidate_sources)
print(len(result.confirmed))
```

If you want, I can also add a small executable example script under
`examples/` that mirrors this guide.
