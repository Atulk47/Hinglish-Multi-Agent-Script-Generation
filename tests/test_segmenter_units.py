"""Scenes follow the story: author paragraphs are merged when the story does not change, never more than the cap, and a short input stays short."""
import re

from preprocessing.segmenter import SceneSegmenter, unit_gap_scores, pick_unit_starts, split_sentences


def seg(text, max_scenes=15):
    return SceneSegmenter(use_llm=False, max_scenes=max_scenes).segment(text)


DIALOGUE = "\n\n".join(f"राम ने कहा—बात नंबर {i} सुनो भाई, यह बहुत ज़रूरी है।" if i % 2 == 0 else f"श्याम बोला—हाँ राम, मैं बात नंबर {i} सुन रहा हूँ।" for i in range(24))


def test_a_long_exchange_in_one_place_is_not_one_scene_per_line_and_respects_the_cap():
    r = seg(DIALOGUE)
    assert 1 <= len(r.scenes) <= 15
    assert len(r.scenes) < 24                                                         # one paragraph per scene is exactly what this replaces


def test_a_short_input_gets_few_scenes_not_a_forced_number():
    short = "सीमा ने चाय बनाई। वह खिड़की के पास बैठ गई।\n\nरवि आया और उसने कहा—चाय मिलेगी? सीमा मुस्कराई।\n\nदोनों ने साथ चाय पी।"
    assert len(seg(short).scenes) <= 2


def test_time_and_place_changes_open_new_scenes():
    story = ("\n\n".join(
        [f"सुबह का समय था, राम घर पर काम कर रहा था। उसने खाना खाया और अख़बार पढ़ा। फिर वह बैठा रहा {i}।" if i == 0 else
         f"अगले दिन राम बाज़ार गया और उसने कई चीज़ें देखीं। दुकानदार से बात की और घर लौट आया {i}।" if i == 1 else
         f"कुछ दिनों बाद राम स्कूल पहुँचकर बच्चों से मिला। उसने सबको मिठाई बाँटी और हँसकर बात की {i}।" for i in range(3)]))
    assert len(seg(story).scenes) == 3


def test_never_more_than_the_cap_even_when_every_paragraph_is_a_new_scene():
    paras = [f"अगले दिन गाँव {i} में सुबह हुई। लोग अपने काम पर निकल पड़े। बच्चे खेलने लगे और बुज़ुर्ग बातें करने लगे।" for i in range(40)]
    for cap in (15, 8):
        assert len(seg("\n\n".join(paras), max_scenes=cap).scenes) <= cap


def test_no_tiny_scene_unless_the_story_is_tiny():
    r = seg(DIALOGUE)
    assert all(len(split_sentences(sc)) >= 3 for sc in r.scenes)


def test_arrival_and_return_cues_start_a_new_scene():
    units = [split_sentences("हामिद ने चिमटा खरीदा। उसे बहुत खुशी हुई। वह दुकान से हटा।"),
             split_sentences("घर पहुँचते ही अमीना ने उसे देखा। उसके हाथ में चिमटा था। वह हैरान रह गई।")]
    assert unit_gap_scores(units)[0] >= 0.33


def test_scenes_are_built_from_whole_paragraphs_and_cover_the_text():
    text = "\n\n".join(["राम घर में था। उसने पानी पिया। फिर वह चुप बैठा रहा।", "अगले दिन वह बाज़ार गया। वहाँ उसने फल देखे। दाम पूछे।", "रात को वह लौटा। उसने खाना खाया। फिर सो गया।"])
    r = seg(text)
    joined = " ".join(r.scenes)
    assert re.sub(r"\s+", "", joined) == re.sub(r"\s+", "", text)
    assert r.starts[0] == 0 and r.starts == sorted(set(r.starts))


def test_a_strong_place_cue_inside_a_paragraph_starts_a_new_scene_when_both_sides_are_long_enough():
    from preprocessing.segmenter import cue_cuts
    ss = split_sentences("दोनों भाई घर की ओर चल पड़े। बारिश तेज़ थी। मोहन भीग गया। घर पहुँचकर माँ ने दरवाज़ा खोला। माँ नाराज़ हुई। फिर वह नरम पड़ गई।")
    assert cue_cuts(ss) == [3]
    assert cue_cuts(ss[:4]) == []                                              # the right side would have a single sentence


def test_the_size_ceiling_follows_the_story_and_the_cap():
    from preprocessing.segmenter import scene_size_ceiling
    assert scene_size_ceiling(16, 15) == 16 and scene_size_ceiling(90, 15) == 16
    assert scene_size_ceiling(300, 15) == 30                                   # a long story can only fit in 15 scenes if they are longer
    assert scene_size_ceiling(300, 8) > 30


def test_a_merged_group_longer_than_the_ceiling_is_cut_with_sentence_level_signals_and_the_cap_still_holds():
    one_paragraph = " ".join(f"राम ने काम {i} किया और आगे बढ़ा।" if i % 7 else f"अगले दिन राम {i} जगह पहुँचा और रुका।" for i in range(40))
    r = seg(one_paragraph)
    assert 2 <= len(r.scenes) <= 15


BENCH = ("स्कूल की छुट्टी हो चुकी थी और आरव और कबीर बेंच पर बैठे थे। वे गणित की परीक्षा की बात कर रहे थे। कबीर बार-बार कह रहा था कि उसने पूरी तैयारी कर ली है।\n\n"
         "आरव ने उसकी किताब देखी और हँसने लगा। कबीर ने किताब बंद कर दी। उसने कहा कि वह बस आराम कर रहा था।\n\n"
         "आरव ने उसे याद दिलाया कि पिछली परीक्षा में भी उसने यही कहा था। कबीर कुछ पल चुप रहा। फिर उसने पूछा कि आरव ने कठिन सवाल देखे हैं या नहीं।\n\n"
         "कबीर ने मज़ाक उड़ाते हुए कहा कि आरव हर चीज़ को इतना गंभीर क्यों लेता है। आरव ने जवाब दिया कि वह हार नहीं मानता। दोनों हँसते रहे।\n\n"
         "फिर कबीर ने किताब खोली और कहा कि वह सवाल समझा दे। आरव ने सिर हिलाया। दोनों साथ किताब देखने लगे।")


def test_one_continuous_conversation_in_one_place_stays_one_scene():
    r = seg(BENCH)
    assert len(r.scenes) == 1


def test_connectives_are_not_scene_changes_but_a_real_time_shift_still_is():
    from preprocessing.segmenter import _cue_score
    assert _cue_score("फिर कबीर ने किताब खोली।") < 0.2 and _cue_score("अगले दिन कबीर स्कूल गया।") == 1.0


def test_reported_speech_counts_as_speech_when_grouping_paragraphs():
    from preprocessing.segmenter import _speech
    assert _speech("कबीर ने मज़ाक उड़ाते हुए कहा कि आरव हर चीज़ को इतना गंभीर क्यों लेता है।") and not _speech("शाम का समय था और बादल घिर आए थे।")


def test_chunks_are_exactly_the_segmenters_scenes_even_for_a_one_scene_story():
    from preprocessing.chunker import ProseChunker
    one = " ".join(f"यह वाक्य संख्या {i} है और बहुत लंबा भी है ताकि शब्द गिनती बढ़े।" for i in range(14))
    assert len(ProseChunker().from_scenes([one])) == 1                              # chunk() would have cut it into ~4-sentence groups
    assert len(ProseChunker().chunk(one)) > 1                                        # (the old behaviour, which is why from_scenes exists)
    c = ProseChunker().from_scenes(["पहला दृश्य।", "", "दूसरा दृश्य।"])
    assert [x["scene_id"] for x in c] == ["scene_01", "scene_02"] and set(c[0]) >= {"scene_id", "text", "has_location_cue", "has_temporal_cue"}


def test_a_paragraph_opening_with_a_time_of_day_is_a_time_jump_even_in_the_middle_of_a_conversation():
    units = [split_sentences("रिया ने कहा कि अब यह मुश्किल नहीं लग रहा। सिया ने कहा कि हाँ, मिलकर करेंगे। दोनों ने फिर से मॉडल देखा।"),
             split_sentences("शाम होने तक मॉडल का ढाँचा बन चुका था। दोनों ने सामान समेटा। रिया ने कहा कि कल फिर मिलेंगे।")]
    assert unit_gap_scores(units)[0] >= 0.33                                   # both paragraphs are speech, but the time jump wins
    same = [units[0], split_sentences("सिया ने कहा कि वह मदद करेगी। रिया ने कहा कि ठीक है। दोनों काम में लग गईं।")]
    assert unit_gap_scores(same)[0] < 0.33                                     # without the time jump the conversation continues


def test_the_science_model_story_gets_the_evening_as_its_own_scene():
    r = seg(open("data/raw/heldout_science_model.txt", encoding="utf-8").read())
    assert 3 <= len(r.scenes) <= 4 and r.scenes[-1].startswith("शाम होने तक")


def test_an_llm_referee_far_from_the_signal_proposal_is_ignored_so_the_scene_count_is_stable():
    """Idgah gave 9 scenes in one run and 15 in the next because the referee reshuffled everything: its answer must stay near the deterministic proposal."""
    from types import SimpleNamespace
    paras = [f"अगले दिन राम {i} जगह गया। वहाँ उसने कई चीज़ें देखीं। फिर वह लौट आया।" if i % 2 == 0 else f"राम ने कहा कि वह {i} बार गया। श्याम ने कहा कि ठीक है। दोनों हँसे।" for i in range(12)]
    text = "\n\n".join(paras)
    det = SceneSegmenter(use_llm=False).segment(text)

    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"scene_starts": [' + ",".join(str(i + 1) for i in range(12)) + ']}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    r = SceneSegmenter(client=client, use_llm=True).segment(text)
    assert len(r.scenes) == len(det.scenes) and "llm" not in r.method and "rejected" in r.notes
