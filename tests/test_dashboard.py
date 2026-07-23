"""The dashboard's model load must be as guarded as the API's.

Regression for the audit finding that the dashboard used a bare ``joblib.load``
catching only ``FileNotFoundError``: a present-but-cross-version artefact
bypassed the version guard and surfaced as a raw ``ModelVersionError`` traceback
at predict time, instead of the clean "unavailable" message the API returns.
``load_model`` now loads through ``get_regressor``, so a rejected artefact is
handled the same way on both surfaces.
"""

from __future__ import annotations


def test_dashboard_load_is_guarded_against_cross_version(monkeypatch) -> None:
    import src.models.predict as pred_mod
    import streamlit_app.app as app

    def _refuse(*args, **kwargs):
        raise pred_mod.ModelVersionError("cross-version artefact")

    monkeypatch.setattr(pred_mod, "get_regressor", _refuse)
    app.load_model.clear()  # drop any cached model so the patched loader runs
    assert app.load_model() is None
