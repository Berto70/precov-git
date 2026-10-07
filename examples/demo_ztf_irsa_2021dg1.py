"""
Fetch a small, real ZTF difference-image sample along the path of 2021 DG1 from
IRSA and check that `ZTFDifferenceImageProvider` can ingest it. 2021 DG1 was
discovered in Pan-STARRS images.

Steps
-----
1. Sky position of the target for EACH night (JPL Horizons, 1-day steps).
2. Per night, IRSA/IBE science-exposure metadata for quadrants overlapping a box
   around that night's track (POS/SIZE/WHERE/COLUMNS/ct API). If the server-side
   time filter times out, the query is retried as a spatial-only query and the
   time cut is applied locally.
3. Keep only quadrants that (approximately) CONTAIN the target at the time of each
   exposure -- overlapping the search box is not enough, the object moves ~1-2 deg/day.
4. Build `*_scimrefdiffimg.fits.fz` URLs, keep those that exist, write a manifest,
   build the provider, download one image and print its metadata.

You can skip the IRSA query with --metadata-csv FILE (a CSV saved from the browser
using the query URL this script prints); steps 3-4 then run on that file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.time import Time

from precov.survey import ZTFDifferenceImageProvider
from precov.survey import horizons, irsa


def make_position_interpolator(track):
    """Linear interpolation of the Horizons track: jd -> (ra_deg, dec_deg)."""
    jd = np.array([t[0] for t in track])
    ra = np.unwrap(np.radians([t[1] for t in track]))
    dec = np.array([t[2] for t in track])

    def position_at(j: float):
        return float(np.degrees(np.interp(j, jd, ra)) % 360.0), float(np.interp(j, jd, dec))

    return position_at


def finish(selected, args):
    """Manifest -> provider -> download one image -> print metadata."""
    manifest = irsa.write_manifest(selected, Path(args.out_manifest), maglim_sigma=args.maglim_sigma)
    print(f"\nWrote manifest with {len(selected)} images: {manifest}")
    print("Building provider (downloads the files on first access) ...")
    provider = ZTFDifferenceImageProvider.from_manifest(manifest, cache_dir=args.cache_dir)
    images = provider.list_images()
    print(f"Provider images: {len(images)}")
    for img in images[:3]:
        print(f"  {img.image_id}: shape={img.shape}, t={img.t.isot}, "
              f"lim3={img.limiting_mag_3sigma:.2f}, fwhm_px={img.fwhm_px:.2f}")
    sample = images[0]
    data = provider.get_image_data(sample.image_id)
    print(f"Loaded pixels for {sample.image_id}: {data.shape}, nanmean={float(np.nanmean(data)):.4f}")
    print("Done. The manifest/provider can now be used in PrediscoveryPipeline.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="2021 DG1")
    ap.add_argument("--jd-min", type=float, default=2459246.5, help="default 2021-02-01 00:00 UT")
    ap.add_argument("--jd-max", type=float, default=2459274.5, help="default 2021-03-01 00:00 UT")
    ap.add_argument("--min-dec", type=float, default=-32.0,
                    help="ZTF (Palomar) does not observe much below Dec ~ -30; nights below this are skipped")
    ap.add_argument("--ephemeris-only", action="store_true",
                    help="print the per-night ephemeris and ZTF-reachability, then stop")
    ap.add_argument("--metadata-csv", help="use this IRSA metadata CSV instead of querying IRSA")
    ap.add_argument("--size-deg", type=float, default=0.5,
                    help="base search box (full width); widened automatically for fast movers")
    ap.add_argument("--footprint-margin", type=float, default=0.05,
                    help="extra deg around the nominal quadrant when testing 'contains the target'")
    ap.add_argument("--per-night", type=int, default=1, help="max quadrants kept per night")
    ap.add_argument("--max-images", type=int, default=8, help="total cap (each image is ~10-40 MB)")
    ap.add_argument("--no-probe", action="store_true", help="skip URL existence checks")
    ap.add_argument("--maglim-sigma", type=float, default=5.0,
                    help="sigma level that IRSA 'maglimit' corresponds to (converted to 3 sigma)")
    ap.add_argument("--out-manifest", default="data/ztf_2021dg1_manifest.csv")
    ap.add_argument("--cache-dir", default="data/ztf_cache")
    args = ap.parse_args()

    # ------------------------------------------------------------------ CSV mode
    if args.metadata_csv:
        rows = irsa.load_metadata_csv(args.metadata_csv)
        print(f"Loaded {len(rows)} metadata rows from {args.metadata_csv}")
        jds = [float(r["obsjd"]) for r in rows]
        track = horizons.daily_positions(args.target, min(jds) - 1.0, max(jds) + 1.0)
        pos_at = make_position_interpolator(track)
        quads = irsa.unique_quadrants(rows)
        inside = irsa.rows_containing(quads, pos_at, args.footprint_margin)
        print(f"Quadrants: {len(quads)}; containing the target at exposure time: {len(inside)}")
        if not inside:
            raise SystemExit(
                "None of these quadrants contains the target (they only overlap the search box). "
                "Try another night, or widen --footprint-margin to test the edge cases."
            )
        selected, failures = irsa.select_available(inside, args.max_images, probe=not args.no_probe)
        if failures:
            print(f"  skipped: {dict(failures)}")
        if not selected:
            raise SystemExit("No difference images available for these quadrants (see reasons above).")
        finish(selected, args)
        return

    # ---------------------------------------------------------- ephemeris per night
    # One point per UT date at 08:00 UT = middle of the Palomar night.
    n_nights = int(round(args.jd_max - args.jd_min))
    jd_first_mid = args.jd_min + 8.0 / 24.0
    print(f"Querying Horizons for {args.target}: {n_nights} nights from JD {args.jd_min} ...")
    track = horizons.daily_positions(args.target, jd_first_mid, jd_first_mid + n_nights - 1)
    pos_at = make_position_interpolator(track)

    def rate_at(k):
        _, ra, dec = track[k]
        _, ra2, dec2 = track[k + 1] if k + 1 < len(track) else track[k - 1]
        return ((ra2 - ra) ** 2 * np.cos(np.radians(dec)) ** 2 + (dec2 - dec) ** 2) ** 0.5

    print(f"\n{'UT date':<11} {'RA':>9} {'Dec':>8} {'deg/day':>8}  ZTF reachable?")
    for k, (_, ra, dec) in enumerate(track):
        date = Time(args.jd_min + k, format="jd").strftime("%Y-%m-%d")
        ok = dec >= args.min_dec
        print(f"{date:<11} {ra:9.4f} {dec:8.3f} {rate_at(k):8.2f}  {'yes' if ok else 'no (Dec too far south)'}")
    nights = [k for k, (_, _, dec) in enumerate(track) if dec >= args.min_dec]
    print(f"\n{len(nights)} of {len(track)} nights have the target in ZTF's declination range.")
    if args.ephemeris_only:
        return
    if not nights:
        raise SystemExit("The target is never north of --min-dec in this window. Choose other dates "
                         "(use --ephemeris-only with a different --jd-min/--jd-max).")

    # ------------------------------------------------------------- query each night
    selected_all, failed_nights, no_hit_nights = [], [], []
    for k in nights:
        if len(selected_all) >= args.max_images:
            break
        _, ra, dec = track[k]
        night_min, night_max = args.jd_min + k, args.jd_min + k + 1.0
        rate = rate_at(k)
        size = args.size_deg + 0.5 * rate
        print(f"\n[night {k + 1}/{len(track)}] JD {night_min:.1f}: RA/Dec = {ra:.4f}, {dec:.4f}  "
              f"(motion {rate:.2f} deg/day, search box {size:.2f} deg)")

        rows = None
        for server_filter in (True, False):
            try:
                url, rows = irsa.query_sci_metadata(ra, dec, size, night_min, night_max,
                                                    server_time_filter=server_filter)
                print(f"  query ({'server' if server_filter else 'local'} time filter): {url}")
                break
            except RuntimeError as exc:
                first_line = str(exc).splitlines()[0]
                print(f"  {'server' if server_filter else 'local'}-time-filter query failed: {first_line}")
        if rows is None:
            failed_nights.append(k + 1)
            continue

        quads = irsa.unique_quadrants(rows)
        inside = irsa.rows_containing(quads, pos_at, args.footprint_margin)
        print(f"  rows: {len(rows)}, unique quadrants: {len(quads)}, containing the target: {len(inside)}")
        if not inside:
            if quads:
                no_hit_nights.append(k + 1)
            continue
        room = min(args.per_night, args.max_images - len(selected_all))
        sel, failures = irsa.select_available(inside, room, probe=not args.no_probe)
        if failures:
            print(f"  skipped: {dict(failures)}")
        selected_all.extend(sel)

    if failed_nights:
        print(f"\nNights where IRSA failed even after the fallback: {failed_nights} (re-run, or use --metadata-csv)")
    if no_hit_nights:
        print(f"Nights with ZTF exposures nearby but none containing the target: {no_hit_nights}")
    if not selected_all:
        raise SystemExit(
            "No difference images found. 'containing the target: 0' = ZTF images only overlapped the "
            "search box; 404 = no reference image at processing time; 401/403 = proprietary data."
        )
    finish(selected_all, args)


if __name__ == "__main__":
    main()
