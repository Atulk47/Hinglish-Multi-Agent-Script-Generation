from types import SimpleNamespace

from evaluation.ablation import run_editor_variant


class _Gate:
    def __init__(self, fail_gate=1):
        self.density_threshold = 0.25
        self._permissive = False
        self._scene_slang_counts = {}
        self.calls = []
        self.fail_gate = fail_gate

    def gate1_speaker_eligible(self, speaker):
        self.calls.append("gate1")
        return self.fail_gate != 1

    def gate2_pragmatic_eligible(self, utterance, intent_classifier):
        self.calls.append("gate2")
        return self.fail_gate != 2, intent_classifier.classify(utterance)

    def gate3_density_ok(self, scene_id, utterance_token_count):
        self.calls.append("gate3")
        self._scene_slang_counts.setdefault(scene_id, {"total": 0, "slang": 0})
        self._scene_slang_counts[scene_id]["total"] += utterance_token_count
        return self.fail_gate != 3

    def run_all_gates(self, speaker, utterance, scene_id, intent_classifier):
        if not self.gate1_speaker_eligible(speaker):
            return {"go": False, "gate_stopped": 1, "intent": None,
                    "reason": "blocked by Gate 1"}
        eligible, intent = self.gate2_pragmatic_eligible(utterance, intent_classifier)
        if not eligible:
            return {"go": False, "gate_stopped": 2, "intent": intent,
                    "reason": "blocked by Gate 2"}
        if not self.gate3_density_ok(scene_id, len(utterance.split())):
            return {"go": False, "gate_stopped": 3, "intent": intent,
                    "reason": "blocked by Gate 3"}
        return {"go": True, "gate_stopped": None, "intent": intent,
                "reason": "passed"}

    def set_permissive(self, enabled):
        self._permissive = bool(enabled)


class _Editor:
    def __init__(self):
        self.gate = _Gate()
        self.intent_clf = SimpleNamespace(classify=lambda text: "CASUAL_ASSERTION")
        self._used = {}
        self._last = None
        self._rng = None
        self.reset_count = 0
        self.last_gate_go = None
        self.last_gate_result = None
        self.last_rule_gate_ok = None

    def reset(self):
        self.reset_count += 1
        self.gate._scene_slang_counts = {}
        self._used = {}
        self._last = None

    def _judge_ok(self, neutral, edited, slang):
        return False, "blocked by original rule gate"

    def rewrite_script(self, neutral_script):
        output = []
        for scene in neutral_script:
            self.last_gate_result = self.gate.run_all_gates(
                "Arav", "This is a neutral sentence", scene["scene_id"], self.intent_clf
            )
            self.last_gate_go = self.last_gate_result["go"]
            self.last_rule_gate_ok = self._judge_ok("original", "edited", "mast")[0]
            output.append({"scene_id": scene["scene_id"], "dialogue": scene["dialogue"],
                           "decisions": []})
        return output


class _Validator:
    def __init__(self, fail_once=False):
        self.calls = 0
        self.fail_once = fail_once

    def diagnose_scene(self, neutral, slang, profiles):
        self.calls += 1
        failed = self.fail_once and self.calls == 1
        return {"passed": not failed,
                "failure_mode": "LOW_COVERAGE" if failed else "OK"}


def _neutral():
    return [{"scene_id": "scene_01", "dialogue": "Arav: Namaste", "events": {}}]


def test_eligibility_gate_ablation_bypasses_and_restores_gates():
    editor = _Editor()
    run_editor_variant(editor, _neutral(), {}, "no_eligibility_gates", _Validator())
    assert editor.last_gate_go is True
    assert editor.gate.run_all_gates("Arav", "text", "scene_01", None)["go"] is False


def test_acceptance_rule_gate_ablation_bypasses_and_restores_rule_gate():
    editor = _Editor()
    run_editor_variant(editor, _neutral(), {}, "no_acceptance_rule_gate", _Validator())
    assert editor.last_rule_gate_ok is True
    assert editor._judge_ok("original", "edited", "mast")[0] is False


def test_no_validation_redo_is_single_pass_and_does_not_call_validator():
    editor = _Editor()
    validator = _Validator()
    slang, diagnostics = run_editor_variant(
        editor, _neutral(), {}, "no_validation_redo", validator
    )
    assert validator.calls == 0
    assert diagnostics == {}
    assert slang[0]["scene_id"] == "scene_01"


def test_full_variant_retries_a_failed_scene_and_revalidates():
    editor = _Editor()
    validator = _Validator(fail_once=True)
    slang, diagnostics = run_editor_variant(
        editor, _neutral(), {}, "full", validator, max_attempts=2
    )
    assert validator.calls == 2
    assert diagnostics["scene_01"]["passed"] is True
    assert slang[0]["events"] == {}


def test_no_editor_keeps_neutral_dialogue_unchanged():
    editor = _Editor()
    slang, diagnostics = run_editor_variant(editor, _neutral(), {}, "no_editor")
    assert slang[0]["dialogue"] == "Arav: Namaste"
    assert slang[0]["decisions"] == []
    assert diagnostics == {}
    assert editor.reset_count == 0


def test_gate1_ablation_only_bypasses_speaker_gate():
    editor = _Editor()
    run_editor_variant(editor, _neutral(), {}, "no_gate1", _Validator())
    assert editor.last_gate_result["go"] is True
    assert editor.gate.calls == ["gate2", "gate3"]


def test_gate2_ablation_only_bypasses_pragmatic_gate():
    editor = _Editor()
    editor.gate = _Gate(fail_gate=2)
    run_editor_variant(editor, _neutral(), {}, "no_gate2", _Validator())
    assert editor.last_gate_result["go"] is True
    # Gate-2 intent classification is still called; its eligibility verdict is overridden.
    assert editor.gate.calls == ["gate1", "gate2", "gate3"]


def test_gate3_ablation_allows_density_but_keeps_token_accounting():
    editor = _Editor()
    editor.gate = _Gate(fail_gate=3)
    run_editor_variant(editor, _neutral(), {}, "no_gate3", _Validator())
    assert editor.last_gate_result["go"] is True
    assert editor.gate.calls == ["gate1", "gate2", "gate3"]
    assert editor.gate._scene_slang_counts["scene_01"]["total"] > 0
