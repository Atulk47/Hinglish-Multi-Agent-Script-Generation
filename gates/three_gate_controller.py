from memory.scratchpad import Scratchpad


class ThreeGateController:
    """
    Controls slang insertion eligibility through 3 sequential gates.

    Gate 1: Speaker Eligibility  — blocks elders/authority figures
    Gate 2: Pragmatic Eligibility — blocks formal/neutral intents
    Gate 3: Density Control       — blocks if scene already has too much slang
    """

    def __init__(self, scratchpad: Scratchpad,
                  slang_density_threshold: float = 0.25):
        self.scratchpad = scratchpad
        self.density_threshold = slang_density_threshold
        # {scene_id: {"total": int, "slang": int}}
        self._scene_slang_counts = {}

    # ── GATE 1: Speaker Eligibility ───────────────────────────────────
    def gate1_speaker_eligible(self, speaker_name: str) -> bool:
        """
        Returns False if this speaker should NEVER receive slang.
        Elders and authority figures always return False.
        """
        profiles = self.scratchpad.read("character_profiles") or {}

        # Case-insensitive lookup
        profile = {}
        for key in profiles:
            if key.lower() == speaker_name.lower():
                profile = profiles[key]
                break

        role = profile.get("social_role", "peer")
        slang_level = profile.get("slang_level", 5)

        if role in ["elder", "authority"]:
            return False

        if slang_level == 0:
            return False

        return True

    # ── GATE 2: Pragmatic Eligibility ─────────────────────────────────
    def gate2_pragmatic_eligible(self, utterance: str,
                                   intent_classifier) -> tuple:
        """
        Returns (eligible: bool, intent_label: str).
        Only permits slang for casual/emotional intents.
        """
        SLANG_PERMITTED_INTENTS = {
            "REQUEST_URGENCY",
            "ASSERT_FRUSTRATION",
            "ASSERT_POSITIVE",
            "EXPRESS_STRESS",
            "CASUAL_ASSERTION"
        }
        SLANG_BLOCKED_INTENTS = {
            "INFORM_NEUTRAL",
            "ASSERT_NEUTRAL",
            "QUESTION_FORMAL"
        }

        intent = intent_classifier.classify(utterance)
        eligible = intent in SLANG_PERMITTED_INTENTS
        return eligible, intent

    # ── GATE 3: Density Control ────────────────────────────────────────
    def gate3_density_ok(self, scene_id: str,
                          utterance_token_count: int) -> bool:
        """
        Returns False if this scene has already reached the slang density budget.
        Density = slang_tokens / total_tokens. Must stay below threshold (default 0.25).
        """
        if scene_id not in self._scene_slang_counts:
            self._scene_slang_counts[scene_id] = {"total": 0, "slang": 0}

        counts = self._scene_slang_counts[scene_id]
        counts["total"] += utterance_token_count

        if counts["total"] == 0:
            return True

        current_density = counts["slang"] / counts["total"]
        return current_density < self.density_threshold

    def record_slang_insertion(self, scene_id: str, slang_token_count: int):
        """Call this after a slang phrase is successfully inserted."""
        if scene_id not in self._scene_slang_counts:
            self._scene_slang_counts[scene_id] = {"total": 0, "slang": 0}
        self._scene_slang_counts[scene_id]["slang"] += slang_token_count

    def run_all_gates(self, speaker: str, utterance: str,
                       scene_id: str, intent_classifier) -> dict:
        """
        Run all 3 gates sequentially.
        Stops at the first failing gate.
        Returns: {"go": bool, "gate_stopped": int|None, "intent": str|None}
        """
        # Gate 1
        g1 = self.gate1_speaker_eligible(speaker)
        if not g1:
            return {"go": False, "gate_stopped": 1, "intent": None,
                    "reason": f"Speaker '{speaker}' is elder/authority or slang_level=0"}

        # Gate 2
        g2, intent = self.gate2_pragmatic_eligible(utterance, intent_classifier)
        if not g2:
            return {"go": False, "gate_stopped": 2, "intent": intent,
                    "reason": f"Intent '{intent}' does not permit slang"}

        # Gate 3
        token_count = len(utterance.split())
        g3 = self.gate3_density_ok(scene_id, token_count)
        if not g3:
            return {"go": False, "gate_stopped": 3, "intent": intent,
                    "reason": f"Scene '{scene_id}' has reached slang density budget"}

        return {"go": True, "gate_stopped": None, "intent": intent,
                "reason": "All gates passed"}

    def get_density_report(self) -> dict:
        """Returns current slang density for all scenes — useful for debugging."""
        report = {}
        for scene_id, counts in self._scene_slang_counts.items():
            total = counts["total"]
            slang = counts["slang"]
            density = slang / total if total > 0 else 0.0
            report[scene_id] = {
                "total_tokens": total,
                "slang_tokens": slang,
                "density": round(density, 3),
                "under_budget": density < self.density_threshold
            }
        return report