"""Artifact-contract tests, the SHIPPED model artefacts must serve correct zone labels.

These tests load the real ``models/*.joblib`` artefacts, which are committed
and MANIFEST-pinned, and probe the CAPABILITY:
the decoded zone label must be the bucket the predicted price falls into
(via ``PRICE_ZONE_BINS``/``zone_for_price``), so zone and price always agree.
The prior suite asserted only label MEMBERSHIP, which stayed green while the API
decoded 3 of the 4 classes wrong (a $1.4M Manhattan condo served "Low").
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import (
    MODELS_DIR,
    PRICE_ZONE_BINS,
    PRICE_ZONE_LABELS,
)


def _models_present() -> bool:
    return all(
        (Path(MODELS_DIR) / name).exists()
        for name in ("price_regressor_best.joblib", "price_interval.json")
    )


pytestmark = pytest.mark.skipif(
    not _models_present(),
    reason="serving artifacts missing (partial checkout), they are committed and pinned by models/MANIFEST.sha256",
)


@pytest.fixture(scope="module")
def artifacts() -> dict:
    """The shipped model."""
    return {"reg": joblib.load(MODELS_DIR / "price_regressor_best.joblib")}


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


def test_the_served_zone_is_the_served_price_bucketed(artifacts: dict) -> None:
    """End-to-end: the API's two fields must agree with each other.

    This is what the single-model architecture buys. With a separate
    classifier the zone came from one model and the price from another, so a
    response could carry zone="High" beside a price that buckets to "Medium"
    and nothing in the system would notice. Here the zone IS the bucketed
    price, and this probes the wired path rather than the library function --
    it fails if the endpoint stops routing through the shared decode.
    """
    from api.main import app
    from src.models.decode import zone_for_price

    client = TestClient(app)

    for name, probe in PROBES.items():
        resp = client.post("/predict", json=_api_payload(probe["row"]))
        assert resp.status_code == 200, f"{name}: {resp.text}"
        body = resp.json()
        assert body["zone"]["price_zone"] == zone_for_price(
            body["price"]["predicted_price"]
        ), name
