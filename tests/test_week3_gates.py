import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from memory.scratchpad import Scratchpad
from gates.three_gate_controller import ThreeGateController


def test_gate1_blocks_elder():
    sp = Scratchpad("test_gate1")
    # Write a profile with social_role = "elder"
    sp.write("character_profiles", {
        "Pitaji": {
            "name": "Pitaji", "age": 60, "social_role": "elder",
            "slang_level": 0, "relationship": {}, "current_emotion": "neutral",
            "turn_count": 0, "stress_index": 0.0
        }
    })
    gate = ThreeGateController(sp)
    assert gate.gate1_speaker_eligible("Pitaji") == False, \
        "Elder should be blocked by Gate 1"
    print("PASS: Gate 1 blocks elder")


def test_gate1_allows_peer():
    sp = Scratchpad("test_gate1_peer")
    sp.write("character_profiles", {
        "Arav": {
            "name": "Arav", "age": 20, "social_role": "peer",
            "slang_level": 5, "relationship": {}, "current_emotion": "neutral",
            "turn_count": 0, "stress_index": 0.0
        }
    })
    gate = ThreeGateController(sp)
    assert gate.gate1_speaker_eligible("Arav") == True, \
        "Peer should be allowed by Gate 1"
    print("PASS: Gate 1 allows peer")


def test_gate3_density_budget():
    sp = Scratchpad("test_gate3")
    sp.write("character_profiles", {})
    gate = ThreeGateController(sp, slang_density_threshold=0.25)

    scene_id = "scene_01"

    # Utterance 1: 10 tokens, then record 5 slang tokens
    # total=10, slang=0 -> density=0.0 -> True (allowed)
    result1 = gate.gate3_density_ok(scene_id, 10)
    assert result1 == True
    gate.record_slang_insertion(scene_id, 5)
    # state now: total=10, slang=5 -> density=0.5

    # Utterance 2: 10 more tokens
    # total becomes 20, slang=5 -> density=5/20=0.25
    # 0.25 < 0.25 is False -> should be BLOCKED
    result2 = gate.gate3_density_ok(scene_id, 10)
    assert result2 == False, f"Should be blocked at density budget, got {result2}"

    print("PASS: Gate 3 blocks at density budget")


def test_run_all_gates_elder_blocked():
    sp = Scratchpad("test_all_gates")
    sp.write("character_profiles", {
        "Madam": {
            "name": "Madam", "age": 45, "social_role": "authority",
            "slang_level": 0, "relationship": {}, "current_emotion": "neutral",
            "turn_count": 0, "stress_index": 0.0
        }
    })
    gate = ThreeGateController(sp)

    # We don't need a real intent classifier for this test — Gate 1 stops first
    result = gate.run_all_gates("Madam", "please sit down", "scene_01",
                                  intent_classifier=None)
    assert result["go"] == False
    assert result["gate_stopped"] == 1
    print("PASS: run_all_gates stops at Gate 1 for authority figure")


if __name__ == "__main__":
    test_gate1_blocks_elder()
    test_gate1_allows_peer()
    test_gate3_density_budget()
    test_run_all_gates_elder_blocked()
    print("\nAll Gate tests passed!")