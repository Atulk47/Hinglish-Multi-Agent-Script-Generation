# Demo checklist (verified 2026-11-19 through the real UI)

Start: `python -m ui.app` (PowerShell, from the project folder with the `hinglish` env). Models load in ~45-50 s; the page is usable at once, Generate waits for them. Close other heavy apps first (needs ~3 GB free RAM).

Defaults for the demo: Voices = **Standard (free)**, Slang Director = ON (default), AI judge = off, Render audio drama = on.

Verified end to end (Standard voices): `demo_science_fair` (~70 s), `demo_hostel_project_night` (~90 s), `demo_cricket_tournament` (~210 s, 10 scenes). Each: screenplay + slang diff + characters/emotion arcs + audio (slang and neutral mp3) + subtitles + player.html + downloads, no Devanagari in the output, no errors. Cost per story about $0.05 (Director + writer calls).

NOT verified (cost money or wake a paid GPU, so I did not run them): **Studio** voices (~$0.1/min) and **Maya** (MAYA_URL is set: it boots a serverless GPU). Try them once before the demo if you want to use them; Standard is the safe default.

Slang to expect: about 15-17% of lines, ~2 per lively peer scene of 4+ lines; none for elders / teachers / serious or traditional scenes (by design). Pick lively peer stories for the demo.

## Fixed during demo preparation (all general, all tested)
* English loanwords with the candra-o vowel (doctor, college, hostel, office, laptop, warden, order ...) were mangled in the romanisation ("kleja", "hstala", "ऑphisa"): now correct; this is why "वॉर्डन शर्मा सर" had become the first name "Wardhan Sharma".
* "riyayat" (leniency) was misread as "rearrangement": a small glossary of easily misread Hindi/Urdu words (`data/lexicon/hindi_gloss.json`) is given to the writer.
* Markdown asterisks ("* * Tumhari mehnat ...") no longer reach the screenplay or the subtitles.
* The narrator no longer announces "Scene 2. Unknown." and capitalises a location.
* Glued "Schoolmein" is split; "sab bakwaas ja raha hai" is blocked in the slang pool; "bura laga" can no longer become "bore"; "mujhe jhakkas laga ki ..." is rejected; a source word such as "super bro" is not reported as a slang leak.

## Known small things
* The writer is an LLM: two runs of the same story differ (a line may be missing or oddly phrased). Run the story once before the demo and, if you like the result, show that run: the UI saves everything under `outputs/ui_<timestamp>_*`.
* Scene-location announcements come from the story text only; a scene with no clear place gets just "Scene N.".
