"""
Self-consistency check on the set of significant linking results (Sec. 3.6).

A single spurious orbit can, by chance, pass close to isolated artifacts in
several images and be promoted by the significance threshold alone. As a
last check, the paper requires that significant candidates be *mutually*
consistent: for candidate n's refit orbit, record which catalog source in
each destination image m' minimized the Mahalanobis distance (already stored
in `LinkResult.matches`); n is retained only if, for every *other* candidate
whose own image lies among n's matched images, that other candidate's source
is exactly the one n matched to. In other words, the matched sources must
themselves be members of the significant-candidate set, referencing each
other consistently.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..data_models import LinkResult


def verify_mutual_consensus(link_results: List[LinkResult]) -> List[LinkResult]:
    """Filter `link_results` (already thresholded on significance) down to
    the mutually self-consistent subset."""
    # Index every significant candidate by (image_id, source_index).
    index: Dict[Tuple[str, int], LinkResult] = {
        (lr.candidate.image_id, lr.candidate.source_index): lr for lr in link_results
    }

    confirmed = []
    for lr in link_results:
        matched_ids = {(img_id, src_idx) for img_id, (src_idx, _r) in lr.matches.items()}

        # Other significant candidates whose image lr's orbit actually
        # intersected (i.e. appears in lr.matches).
        others_in_range = [
            other for other in link_results
            if other is not lr and other.candidate.image_id in lr.matches
        ]

        if not others_in_range:
            # Nothing to cross-check against; conservatively reject, matching
            # the paper's framing that consensus requires *multiple* mutually
            # -confirming detections, not an isolated significant source.
            continue

        is_consistent = all(
            (other.candidate.image_id, other.candidate.source_index) in matched_ids
            for other in others_in_range
        )
        if is_consistent:
            confirmed.append(lr)

    return confirmed
