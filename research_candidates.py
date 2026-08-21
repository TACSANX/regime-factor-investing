from __future__ import annotations

import copy

import research_backtest as rb


_original_score_variants = rb.score_variants
_original_strategy_specs = rb.strategy_specs


def _zero_many(config: dict, factors: list[str]) -> dict:
    c = copy.deepcopy(config)
    for factor in factors:
        for weights in c["regime_weights"].values():
            weights[factor] = 0.0
        if factor == "macro_sector":
            c["macro_sector_tilts"] = {regime: {} for regime in c["regime_weights"]}
    return c


def _static_neutral_no_growth_low_vol(config: dict) -> dict:
    c = copy.deepcopy(config)
    neutral = dict(c["regime_weights"]["NEUTRAL"])
    neutral["growth"] = 0.0
    neutral["low_vol"] = 0.0
    neutral["macro_sector"] = 0.0
    for regime in c["regime_weights"]:
        c["regime_weights"][regime] = dict(neutral)
    c["macro_sector_tilts"] = {regime: {} for regime in c["regime_weights"]}
    return c


def score_variants(config: dict) -> dict[str, dict]:
    variants = _original_score_variants(config)
    variants["no_growth_low_vol"] = _zero_many(config, ["growth", "low_vol"])
    variants["no_growth_low_vol_macro_sector"] = _zero_many(config, ["growth", "low_vol", "macro_sector"])
    variants["static_neutral_no_growth_low_vol"] = _static_neutral_no_growth_low_vol(config)
    return variants


def strategy_specs() -> list[dict]:
    specs = _original_strategy_specs()
    # Exploratory interaction hypotheses identified before this rerun from the
    # one-factor ablations. They are intentionally labelled as hypotheses and
    # must pass the same bootstrap / rolling / multiple-testing gates as every
    # other variant. They are NOT automatically promoted to the live model.
    specs.extend([
        {
            "strategy": "hyp_no_growth_low_vol_n10",
            "score_variant": "no_growth_low_vol",
            "n": 10,
            "allocation": "invvol",
            "category": "interaction_hypothesis",
        },
        {
            "strategy": "hyp_no_growth_low_vol_macro_sector_n10",
            "score_variant": "no_growth_low_vol_macro_sector",
            "n": 10,
            "allocation": "invvol",
            "category": "interaction_hypothesis",
        },
        {
            "strategy": "hyp_static_neutral_no_growth_low_vol_n10",
            "score_variant": "static_neutral_no_growth_low_vol",
            "n": 10,
            "allocation": "invvol",
            "category": "interaction_hypothesis",
        },
    ])
    return specs


rb.score_variants = score_variants
rb.strategy_specs = strategy_specs


if __name__ == "__main__":
    rb.run(rb.parse_args())
