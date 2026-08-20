import json
import os
import re
import time
from typing import List

import config


# ── Prompt template ────────────────────────────────────────────────────
NARRATIVE_ANALYST_PROMPT = """You are a screenplay structural analyst.
Your ONLY job is to extract a structured event chain from a Hindi story (written in Roman script).
Do NOT generate any dialogue. Only extract what exists.

Story text for THIS SCENE ONLY (scene_id = "{scene_id}"):
{story_text}

Previous scene summary (for continuity):
{previous_scene_summary}

Output a JSON array containing EXACTLY ONE object. No preamble. No explanation. No markdown code fences. Just the raw JSON array with one element.

The single object must have EXACTLY these fields:
{{
  "scene_id": "{scene_id}",
  "key_events": ["Event 1 description", "Event 2 description"],
  "characters": ["CharName1", "CharName2"],
  "scene_goal": "What this scene accomplishes narratively",
  "location": "Where this scene takes place",
  "narrative_link": "Following [previous event], [character] now [does X]",
  "pronoun_map": {{"usne": "Ram", "woh": "Sita"}}
}}

CRITICAL RULES:
- ALWAYS use scene_id = "{scene_id}" exactly — do not create additional scene_ids, do not split this text into multiple scenes.
- Output EXACTLY ONE JSON object inside the array, even if the scene text describes multiple events or covers a time span. Combine everything into ONE object's key_events list.
- "characters" must contain ONLY proper names of individual named people (e.g. "Arav", "Ram", "Shyam"). 
- NEVER include: groups of people (e.g. "log"/people, "chatron"/students, "shodhakartaon"/researchers), organizations or institutions (e.g. "nagar parishad"/city council), objects (e.g. "dayari"/diary), or generic role references to unnamed people (e.g. "bhule-bisare vyapari"/the forgotten merchant) UNLESS that is literally the character's only name given in the story.
- If NO named individual characters appear in this scene, "characters" must be an empty list [].
- key_events: list every plot-significant event (minimum 1, maximum 6)
- narrative_link: MUST reference what happened in the previous scene
- pronoun_map: map every pronoun in this chunk to the character it refers to (use {{}} if none)
"""


class NarrativeAnalyst:
    def __init__(self, llm_backend: str = None):
        # llm_backend: "ollama" | "groq" | "hf"
        self.backend = llm_backend or config.LLM_BACKEND
        self._setup_llm()

    def _setup_llm(self):
        if self.backend == "ollama":
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3:8b-instruct-q4_K_M",
                               temperature=0.1)

        elif self.backend == "groq":
            from groq import Groq
            from dotenv import load_dotenv
            load_dotenv()
            self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

        elif self.backend == "hf":
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct")
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                torch_dtype=torch.float16,
                load_in_4bit=True)
            self.pipe = pipeline("text-generation", model=model,
                                  tokenizer=tokenizer, max_new_tokens=1024)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def extract_event_chain(self, scene_chunks: List[dict],previous_summary: str = "This is the first scene.") -> List[dict]:
        all_events = []

        for chunk in scene_chunks:
            prompt = NARRATIVE_ANALYST_PROMPT.format(
                scene_id=chunk["scene_id"],
                story_text=chunk["text"],
                previous_scene_summary=previous_summary
            )

            raw_output = self._call_llm(prompt)
            events = self._parse_json_output(raw_output, chunk["scene_id"])

            # Guard: if the model returned nothing for this scene, insert a fallback
            if not events:
                events = [{
                    "scene_id": chunk["scene_id"],
                    "key_events": [],
                    "characters": [],
                    "scene_goal": "Unknown",
                    "location": "Unknown",
                    "narrative_link": previous_summary,
                    "pronoun_map": {}
                }]

            # Guard: force every returned object to use the correct scene_id
            # (in case the model still drifts) and take only the FIRST object
            # if multiple were returned despite instructions
            first_event = events[0]
            first_event["scene_id"] = chunk["scene_id"]
            all_events.append(first_event)

            previous_summary = first_event.get("scene_goal", previous_summary)

        return all_events

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "ollama":
            return self.llm.invoke(prompt)

        elif self.backend == "groq":
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.groq_client.chat.completions.create(
                        model=config.GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1500,
                        reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Groq call failed ({e}), retrying in 2s...")
                        time.sleep(2)
                    else:
                        raise

        elif self.backend == "hf":
            result = self.pipe(prompt, return_full_text=False)
            return result[0]["generated_text"]

    def _parse_json_output(self, raw: str, fallback_scene_id: str) -> List[dict]:
        """Extract JSON from LLM output robustly."""
        # Strip markdown code fences if present
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()

        # Find JSON array in output
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if not match:
            # Try wrapping a single object in an array
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = f"[{match.group()}]"
            else:
                # Return fallback structure
                return [{
                    "scene_id": fallback_scene_id,
                    "key_events": [],
                    "characters": [],
                    "scene_goal": "Unknown",
                    "location": "Unknown",
                    "narrative_link": "",
                    "pronoun_map": {}
                }]
        else:
            cleaned = match.group()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to clean common LLM JSON errors (trailing commas)
            clean = re.sub(r',\s*}', '}', cleaned)
            clean = re.sub(r',\s*]', ']', clean)
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                return [{
                    "scene_id": fallback_scene_id,
                    "key_events": [],
                    "characters": [],
                    "scene_goal": "Unknown",
                    "location": "Unknown",
                    "narrative_link": "",
                    "pronoun_map": {}
                }]


if __name__ == "__main__":
    analyst = NarrativeAnalyst(llm_backend="groq")
    test_chunks = [
        {"scene_id": "scene_01", "text": "subah ram apne ghar se nikla aur school ki taraf chal diya. raaste mein usne apne dost shyam ko dekha."}
    ]
    events = analyst.extract_event_chain(test_chunks)
    print(json.dumps(events, indent=2, ensure_ascii=False))