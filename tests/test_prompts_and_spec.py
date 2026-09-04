from __future__ import annotations

from pathlib import Path

import yaml

from prompts.templates import TIER_DEFS, build_consul_prompt, build_r1_prompt, build_r2_prompt


INDICATOR = {
    "real_name": "Infrastructure_quality",
    "alias_name": "Infrastructure_quality",
    "city_def": "Synthetic definition",
    "anchor_tier_map": "S+ -> Anchor A\nBOT -> Anchor B",
}
PACKET = {f"model_{i}": {"City": {"tier": "B", "reasoning": "reason"}} for i in range(8)}


def test_all_three_prompt_builders_and_21_tiers() -> None:
    assert len(TIER_DEFS) == 21
    r1_system, _ = build_r1_prompt(INDICATOR, ["City"])
    r2_system, _ = build_r2_prompt(INDICATOR, ["City"], PACKET)
    consul_system, _ = build_consul_prompt(INDICATOR, ["City"], PACKET, PACKET)
    assert "S+" in r1_system and "BOT" in r1_system
    assert "坚持己见" in r2_system
    assert "你只决定\"删谁\"" in consul_system


def test_model_specifications_match_manuscript_snapshots() -> None:
    path = Path(__file__).parents[1] / "model_specifications.yml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = [item["model"] for item in spec["parliament_members"]]
    assert models == [
        "gpt-5-nano",
        "qwen-flash",
        "doubao-seed-2-0-mini-260215",
        "deepseek-v4-flash",
        "glm-4-flash",
        "gemini-2.5-flash",
        "grok-4-fast-non-reasoning",
        "llama-4-scout-17b-16e-instruct",
    ]
    assert spec["consul"]["model"] == "gemini-2.5-pro"
    assert spec["request_settings"]["max_output_tokens"] == 16000
