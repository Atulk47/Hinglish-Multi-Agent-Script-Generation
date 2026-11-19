from data_prep import v7_assemble as A


def test_split_is_deterministic_and_by_story():
    assert A.split_of("abc123") == A.split_of("abc123")
    parts = {A.split_of(f"s{i}") for i in range(200)}
    assert parts == {"train", "dev", "test"}


def test_added_key_names_new_words_only():
    assert A.added_key("Yeh plan accha hai", "Yeh plan zabardast hai") == "zabardast"
    assert A.added_key("Yeh plan accha hai", "Yeh plan accha hai") == "(reorder)"


def test_hints_are_optional_and_reproducible():
    c = {"id": "c1", "hints_all": ["bindaas", "mast", "kadak", "solid", "epic"]}
    assert A.with_hints(c, 0.0)["hints"] == []
    h = A.with_hints(c, 1.0)["hints"]
    assert len(h) == 4 and h == A.with_hints(c, 1.0)["hints"]
    assert A.with_hints({"id": "c2"}, 1.0)["hints"] == []
