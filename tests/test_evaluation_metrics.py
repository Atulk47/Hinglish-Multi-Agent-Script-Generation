from evaluation.metrics import aggregate, scene_slang_metrics


def test_validity_is_undefined_when_no_slang_was_inserted():
    metrics = scene_slang_metrics("Arav: Yeh kitaab purani hai.",
                                  "Arav: Yeh kitaab purani hai.", set(), set())
    assert metrics["validity"] is None
    summary = aggregate([metrics])
    assert summary["slang_validity"] is None


def test_validity_is_scored_when_new_slang_was_inserted():
    metrics = scene_slang_metrics("Arav: Yeh idea accha hai.",
                                  "Arav: Yeh idea mast hai.", {"mast"}, {"mast"})
    assert metrics["validity"] == 1.0
    assert aggregate([metrics])["slang_validity"] == 1.0
