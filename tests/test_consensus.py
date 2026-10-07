import numpy as np
from astropy.time import Time

from precov.data_models import CandidateSource, LinkResult, OrbitFitResult
from precov.linking.consensus import verify_mutual_consensus


def _dummy_fit():
    return OrbitFitResult(theta=np.zeros(6), epoch=Time("2020-01-01"),
                           covariance=np.eye(6), cost=0.0, n_obs=0, success=True)


def _link_result(image_id, source_index, matches, significance=15.0):
    candidate = CandidateSource(image_id=image_id, source_index=source_index,
                                 xy=(0.0, 0.0), ra=0.0, dec=0.0, sigma_centroid_px=1.0)
    return LinkResult(candidate=candidate, orbit_fit=_dummy_fit(),
                       significance=significance, log_likelihood_sum=0.0, matches=matches)


def test_three_mutually_confirming_candidates_all_survive():
    A = _link_result("img1", 0, matches={"img2": (0, 0.1), "img3": (0, 0.2)})
    B = _link_result("img2", 0, matches={"img1": (0, 0.1), "img3": (0, 0.15)})
    C = _link_result("img3", 0, matches={"img1": (0, 0.2), "img2": (0, 0.15)})

    confirmed = verify_mutual_consensus([A, B, C])
    confirmed_ids = {(r.candidate.image_id, r.candidate.source_index) for r in confirmed}
    assert confirmed_ids == {("img1", 0), ("img2", 0), ("img3", 0)}


def test_spurious_candidate_pointing_to_wrong_source_is_rejected():
    A = _link_result("img1", 0, matches={"img2": (0, 0.1), "img3": (0, 0.2)})
    B = _link_result("img2", 0, matches={"img1": (0, 0.1), "img3": (0, 0.15)})
    C = _link_result("img3", 0, matches={"img1": (0, 0.2), "img2": (0, 0.15)})
    # D is significant on its own (e.g. an artifact whose orbit happens to
    # pass close to *some* source in img1) but points at a different source
    # index (2) than the one A/B/C mutually confirm (0).
    D = _link_result("img4", 5, matches={"img1": (2, 3.0)})

    confirmed = verify_mutual_consensus([A, B, C, D])
    confirmed_ids = {(r.candidate.image_id, r.candidate.source_index) for r in confirmed}
    assert ("img4", 5) not in confirmed_ids
    assert confirmed_ids == {("img1", 0), ("img2", 0), ("img3", 0)}


def test_isolated_significant_candidate_with_no_cross_confirmation_is_rejected():
    """A single significant result whose matched images contain no other
    significant candidate cannot be self-consistently verified -- this
    matches the paper's framing that consensus requires multiple mutually
    -confirming detections."""
    lone = _link_result("img1", 0, matches={"img2": (3, 0.3)})
    confirmed = verify_mutual_consensus([lone])
    assert confirmed == []


def test_empty_input_returns_empty():
    assert verify_mutual_consensus([]) == []
