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
    # Interaction hypotheses were registered before their first rerun.  The two
    # allocation-decomposition variants below were subsequently registered in
    # research_trial_ledger.csv before being added here.  They answer a narrower
    # question: is the apparent improvement from the factor ranking, or from the
    # inverse-volatility sizing that remains even when the low_vol ranking factor
    # is zeroed?  These remain research-only regardless of the headline CAGR.
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
        {
            "strategy": "hyp_no_growth_low_vol_macro_sector_n10_equal",
            "score_variant": "no_growth_low_vol_macro_sector",
            "n": 10,
            "allocation": "equal",
            "category": "allocation_decomposition",
        },
        {
            "strategy": "hyp_no_growth_low_vol_macro_sector_n10_score",
            "score_variant": "no_growth_low_vol_macro_sector",
            "n": 10,
            "allocation": "score",
            "category": "allocation_decomposition",
        },
    ])
    return specs


rb.score_variants = score_variants
rb.strategy_specs = strategy_specs


if __name__ == "__main__":
    rb.run(rb.parse_args())
