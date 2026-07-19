"""Artifact-contract tests — the SHIPPED model artefacts must serve correct zone labels.

These tests load the real ``models/*.joblib`` artefacts, which are committed
and MANIFEST-pinned, and probe the CAPABILITY:
the decoded zone label must be the label encoder's name for the predicted
class index, and it must be consistent with the predicted price. The prior
suite asserted only label MEMBERSHIP, which stayed green while the API
decoded 3 of the 4 classes wrong (a $1.4M Manhattan condo served "Low").
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import (
    CENTRAL_PARK,
    MANHATTAN_CENTER,
    MODELS_DIR,
    PRICE_ZONE_BINS,
    PRICE_ZONE_LABELS,
)
from src.utils.geo import haversine


def _models_present() -> bool:
    return all(
        (Path(MODELS_DIR) / name).exists()
        for name in (
            "price_zone_best.joblib",
            "price_regressor_best.joblib",
            "label_encoder.joblib",
        )
    )


pytestmark = pytest.mark.skipif(
    not _models_present(),
    reason="serving artifacts missing (partial checkout) — they are committed and pinned by models/MANIFEST.sha256",
)


@pytest.fixture(scope="module")
def artifacts() -> dict:
    """The shipped classifier, regressor, and label encoder."""
    return {
        "clf": joblib.load(MODELS_DIR / "price_zone_best.joblib"),
        "reg": joblib.load(MODELS_DIR / "price_regressor_best.joblib"),
        "le": joblib.load(MODELS_DIR / "label_encoder.joblib"),
    }


def _feature_row(
    beds: int,
    bath: float,
    sqft: float,
    borough: str,
    prop_type: str,
    zipcode: str,
    sublocality: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Mirror api.main._build_features for a direct-artefact probe."""
    total_rooms = beds + bath
    return pd.DataFrame(
        [
            {
                "BEDS": beds,
                "BATH": bath,
                "PROPERTYSQFT": float(sqft),
                "TOTAL_ROOMS": total_rooms,
                "BED_BATH_RATIO": beds / max(bath, 1.0),
                "LOG_SQFT": math.log1p(sqft),
                "ROOMS_PER_SQFT": total_rooms / max(sqft, 1.0),
                "DIST_MANHATTAN_CENTER": haversine(lat, lon, *MANHATTAN_CENTER),
                "DIST_CENTRAL_PARK": haversine(lat, lon, *CENTRAL_PARK),
                "DIST_NEAREST_SUBWAY": haversine(lat, lon, *MANHATTAN_CENTER),
                "BOROUGH": borough,
                "TYPE": prop_type,
                "PROPERTY_CATEGORY": "residential",
                "ZIPCODE": zipcode,
                "SUBLOCALITY": sublocality,
            }
        ]
    )


# Canonical probes spanning the price spectrum (beds, bath, sqft, borough,
# type, zip, sublocality, lat, lon) + the API-payload equivalent.
PROBES: dict[str, dict] = {
    "bronx_small_coop": {
        "row": (
            1,
            1.0,
            550,
            "the bronx",
            "co-op",
            "10457",
            "the bronx",
            40.8460,
            -73.9000,
        ),
    },
    "queens_house": {
        "row": (3, 2.0, 1400, "queens", "house", "11361", "queens", 40.7635, -73.7710),
    },
    "manhattan_1br_condo": {
        "row": (
            1,
            1.0,
            750,
            "manhattan",
            "condo",
            "10016",
            "manhattan",
            40.7460,
            -73.9780,
        ),
    },
    "manhattan_2br_condo": {
        "row": (
            2,
            2.0,
            1200,
            "manhattan",
            "condo",
            "10022",
            "midtown east",
            40.7580,
            -73.9680,
        ),
    },
    "manhattan_midtown_lux": {
        "row": (
            3,
            3.0,
            2200,
            "manhattan",
            "condo",
            "10022",
            "manhattan",
            40.7580,
            -73.9720,
        ),
    },
    "manhattan_4br_lux": {
        "row": (
            4,
            4.5,
            3200,
            "manhattan",
            "condo",
            "10013",
            "manhattan",
            40.7190,
            -74.0050,
        ),
    },
}


def _api_payload(row: tuple) -> dict:
    beds, bath, sqft, borough, prop_type, zipcode, sublocality, lat, lon = row
    return {
        "beds": beds,
        "bath": bath,
        "propertysqft": sqft,
        "borough": borough,
        "type": prop_type,
        "zipcode": zipcode,
        "latitude": lat,
        "longitude": lon,
        "sublocality": sublocality,
    }


def _zone_rank(label: str) -> int:
    """Semantic rank of a zone label (Low=0 ... Very High=3)."""
    return PRICE_ZONE_LABELS.index(label)


def _price_bin_rank(price: float) -> int:
    """Rank of the PRICE_ZONE bin the price falls into."""
    return int(np.digitize(price, PRICE_ZONE_BINS[1:-1]))


def test_api_decode_matches_shipped_encoder(artifacts: dict) -> None:
    """/predict must decode exactly le.classes_[argmax], for every probe.

    This pins the serving-path decode order to the shipped label encoder —
    the config-order decode returned 'Medium' for a Bronx probe whose
    correct label is 'Low' and 'Low' for a $1.4M Manhattan condo.
    """
    from api.main import app

    client = TestClient(app)
    le = artifacts["le"]
    clf = artifacts["clf"]

    for name, probe in PROBES.items():
        features = _feature_row(*probe["row"])
        expected_idx = int(np.argmax(clf.predict_proba(features)[0]))
        expected_label = str(le.classes_[expected_idx])

        resp = client.post("/predict", json=_api_payload(probe["row"]))
        assert resp.status_code == 200, f"{name}: {resp.text}"
        zone = resp.json()["zone"]
        assert zone["price_zone"] == expected_label, (
            f"{name}: served {zone['price_zone']!r}, encoder says {expected_label!r}"
        )
        # Probabilities must be keyed in encoder-class order, and the mass
        # reported as `confidence` must sit on the served label.
        assert list(zone["probabilities"]) == [str(c) for c in le.classes_]
        assert zone["probabilities"][zone["price_zone"]] == zone["confidence"]


def test_zone_labels_consistent_with_predicted_prices(artifacts: dict) -> None:
    """Decoded zones must track predicted prices (the capability, not the shape).

    For each probe the served zone's semantic rank must be within one bin of
    the zone implied by the predicted price, luxury probes must never decode
    'Low', sub-$500k probes must never decode 'High'/'Very High', and zone
    rank must correlate strongly with predicted price across the spread.
    Under the config-order decode this fails on all three counts.
    """
    from api.main import app

    client = TestClient(app)

    zone_ranks: list[int] = []
    prices: list[float] = []
    for name, probe in PROBES.items():
        resp = client.post("/predict", json=_api_payload(probe["row"]))
        assert resp.status_code == 200, f"{name}: {resp.text}"
        body = resp.json()
        label = body["zone"]["price_zone"]
        price = body["price"]["predicted_price"]
        zone_ranks.append(_zone_rank(label))
        prices.append(price)

        assert abs(_zone_rank(label) - _price_bin_rank(price)) <= 1, (
            f"{name}: zone {label!r} vs predicted price ${price:,.0f}"
        )
        if price > 2_500_000:
            assert label not in {"Low", "Medium"}, (
                f"{name}: ${price:,.0f} property served zone {label!r}"
            )
        if price < 500_000:
            assert label not in {"High", "Very High"}, (
                f"{name}: ${price:,.0f} property served zone {label!r}"
            )

    # Spearman rank correlation (pearson on ranks; pandas handles ties).
    corr = pd.Series(zone_ranks).rank().corr(pd.Series(prices).rank())
    assert corr >= 0.8, f"zone rank vs price correlation too weak: {corr:.3f}"


def test_predict_module_decode_matches_encoder(artifacts: dict) -> None:
    """src.models.predict must decode through the shipped encoder too."""
    import src.models.predict as pred_mod

    # Reset caches so earlier mock-model tests cannot leak into this one.
    pred_mod._classifier_cache = None
    pred_mod._regressor_cache = None
    pred_mod._label_encoder_cache = None

    le = artifacts["le"]
    clf = artifacts["clf"]
    for name, probe in PROBES.items():
        features = _feature_row(*probe["row"])
        expected = str(le.classes_[int(np.argmax(clf.predict_proba(features)[0]))])
        result = pred_mod.predict_price_zone(features)[0]
        assert result["price_zone"] == expected, f"{name}"
        assert list(result["probabilities"]) == [str(c) for c in le.classes_]
