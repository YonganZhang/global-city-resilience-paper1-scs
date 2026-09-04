"""Twenty-one-point trapezoid calculation used by the GCIRI model.

``cap_abs`` is initial performance loss and ``cap_resp`` is response deficit,
so larger values are worse. ``cap_rec`` is recovery capacity, so larger values
are better. The bundled coefficients are the fixed model parameters used in
the study.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

_V2_COEF_PATH = Path(__file__).with_name("model_parameters") / "giri_v2_coefficients.json"
_V2 = None
_V2_R_GRID = np.linspace(0.0, 1.0, 21)


def _load_v2_coef():
    global _V2
    if _V2 is None:
        _V2 = json.loads(_V2_COEF_PATH.read_text(encoding="utf-8"))
    return _V2


def _ap_ratio(cap_abs: float, cap_resp: float, cap_rec: float) -> float:
    """梯形面积/周长. cap_abs=0 → 0."""
    y = cap_abs
    angle_rad = np.radians(10 + 70 * (cap_rec / 100))
    t_rec = y / np.tan(angle_rad) if np.tan(angle_rad) > 0 else 0.0
    t_resp = cap_resp
    area = 0.5 * (2 * t_resp + t_rec) * y
    side = np.sqrt(y * y + t_rec * t_rec)
    perim = 2 * t_resp + t_rec + y + side
    return area / perim if perim > 0 else 0.0


def compute_trapezoid_giri(F_city: float, cap_resp: float, cap_rec: float, R_Ph: float) -> float:
    """V2 GIRI: 扫 21 点 PC=-A/P → 国家端点 [RR_ymin, 99.8] 归一化 → 真实 R_Ph 处 piecewise-linear 插值.

    Args:
        F_city: ABS 6 indicator 平均 deficit (0-1, higher=worse)
        cap_resp: RESP 6 mean deficit × 100 (0-100, higher=worse)
        cap_rec: REC6 mean × 100 (0-100, higher is better)
        R_Ph: 物理风险 (0-1)
    Returns:
        GCIRI (0-100)
    """
    v2 = _load_v2_coef()
    _V2_FEATS = v2["rr_ymin_regression"]["features"]
    _V2_COEFS = [v2["rr_ymin_regression"]["coefficients"][k] for k in _V2_FEATS]
    _V2_INTERCEPT = v2["rr_ymin_regression"]["intercept"]
    _V2_RR_YMAX = v2["rr_ymax_global"]
    _V2_NPOINTS = v2["n_curve_points"]

    raw = np.empty(_V2_NPOINTS)
    for j, r in enumerate(_V2_R_GRID):
        cap_abs_j = r * 100.0 * (1.0 + F_city) / 2.0
        raw[j] = -_ap_ratio(cap_abs_j, cap_resp, cap_rec)

    raw_PC_ymin = raw[-1]
    # The historical fitted-parameter key is ``cap_rest``; it denotes the
    # recovery-capacity quantity called ``cap_rec`` in the manuscript.
    feat_vals = {"F": F_city, "cap_resp": cap_resp, "cap_rest": cap_rec, "raw_PC_ymin": raw_PC_ymin}
    rr_ymin = _V2_INTERCEPT + sum(_V2_COEFS[i] * feat_vals[k] for i, k in enumerate(_V2_FEATS))

    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin > 1e-12:
        rr_curve = (raw - rmin) / (rmax - rmin) * (_V2_RR_YMAX - rr_ymin) + rr_ymin
    else:
        rr_curve = np.full(_V2_NPOINTS, _V2_RR_YMAX)

    giri = float(np.interp(min(R_Ph, 1.0), _V2_R_GRID, rr_curve))
    return float(np.clip(giri, 0, 100))


def compute_three_capabilities(F_city, cap_resp_mean, cap_rec_mean, r_ph_city):
    """Return loss/deficit ``cap_abs``, ``cap_resp`` and positive ``cap_rec``."""
    R_Ph = min(r_ph_city, 1.0)
    cap_abs = R_Ph * 100 * (1 + F_city) / 2
    return {"F_city": F_city, "R_Ph_city": float(R_Ph),
            "cap_abs": float(cap_abs), "cap_resp": cap_resp_mean * 100,
            "cap_rec": cap_rec_mean * 100}


def compute_city_giri(F_city, cap_resp, cap_rec, r_ph_city):
    """城市级 GCIRI 全 chain: F + 三能力 + 梯形."""
    return compute_trapezoid_giri(F_city, cap_resp, cap_rec, r_ph_city)


if __name__ == "__main__":
    # 5 城 v15 真值测试
    cases = [
        ("Hong Kong", 0.6, 71.67, 58.33, 1.0),
        ("Shenzhen",  0.525, 69.17, 73.75, 0.8721),
        ("Guangzhou", 0.4833, 68.33, 67.5, 0.8251),
        ("London",    0.4667, 48.33, 52.5, 0.1547),
        ("Tokyo",     0.9, 82.5, 87.92, 0.8267),
    ]
    print(f"{'City':<12} F      cap_resp cap_rec R_Ph    GCIRI")
    for name, F, resp, rest, rph in cases:
        g = compute_trapezoid_giri(F, resp, rest, rph)
        print(f"{name:<12} {F:<6} {resp:<8} {rest:<8} {rph:<7} {g:.2f}")
