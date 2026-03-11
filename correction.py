"""
Two-stage transcript correction pipeline (FUZZY-ONLY, no LLM):
  STAGE 1: Rule-based street name correction (fuzzy matching)
  STAGE 2: Rule-based person name correction (fuzzy matching)

Both stages are fully deterministic — no LLM calls, no hallucination risk.

Extracted from the capstone Gradio prototype for integration into the
RAG query pipeline. Only the correction engine is included here — no UI,
no accuracy evaluation, no document loading.

Required dictionary files (JSON):
  - english_words.json   → list of common English words
  - nola_names.json      → {"first_names": [...], "last_names": [...]}
  - nola_streets.json    → list of full street names
"""

import json
import re
from collections import Counter
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List, Tuple


# ============================================================================
# DICTIONARY LOADING
# ============================================================================

def load_dictionaries(
    english_path: str = "english_words.json",
    names_path: str = "nola_names.json",
    streets_path: str = "nola_streets.json",
) -> Dict:
    """
    Load the three specialised dictionaries needed by the correction pipeline.

    Returns a dict with keys:
        english     – set of lowercase common English words
        streets     – list of full street name strings
        names       – combined list of first + last names
        first_names – set of lowercase first names
        last_names  – set of lowercase last names
    """

    # 1. English common words
    try:
        with open(english_path, "r", encoding="utf-8") as f:
            english_words = set(word.lower() for word in json.load(f))
        print(f"[correction] Loaded {len(english_words)} English words")
    except FileNotFoundError:
        print(f"[correction] WARNING: {english_path} not found – using minimal fallback set")
        english_words = {
            "the", "a", "an", "and", "or", "but", "has", "have", "had",
            "was", "were", "been", "is", "are", "weekend", "seconded",
            "representing", "resident", "community", "on", "behalf",
        }

    # 2. New Orleans names
    try:
        with open(names_path, "r", encoding="utf-8") as f:
            names_data = json.load(f)
        first_names = names_data.get("first_names", [])
        last_names = names_data.get("last_names", [])
        names = first_names + last_names
        print(f"[correction] Loaded {len(first_names)} first names and {len(last_names)} last names")
    except FileNotFoundError:
        print(f"[correction] WARNING: {names_path} not found – using minimal fallback set")
        names = ["Moreno", "Palmer", "Cantrell", "Helena"]
        first_names = []
        last_names = []

    # 3. New Orleans streets
    try:
        with open(streets_path, "r", encoding="utf-8") as f:
            streets = json.load(f)
        print(f"[correction] Loaded {len(streets)} street names")
    except FileNotFoundError:
        print(f"[correction] WARNING: {streets_path} not found – using minimal fallback set")
        streets = ["Claiborne Avenue", "Canal Street", "Tchoupitoulas Street"]

    return {
        "english": english_words,
        "streets": streets,
        "names": names,
        "first_names": set(fn.lower() for fn in first_names),
        "last_names": set(ln.lower() for ln in last_names),
    }


# ============================================================================
# PUNCTUATION PRESERVATION HELPER
# ============================================================================

def apply_correction_preserving_punctuation(words_list, position, corrected_value):
    """
    Apply a word-level correction while preserving trailing punctuation
    and possessive markers ('s / s').
    """
    if position >= len(words_list):
        return words_list

    original_word = words_list[position]

    # Detect possessive suffix
    possessive = ""
    temp_word = original_word
    if temp_word.endswith("'s") or temp_word.endswith("\u2019s"):
        possessive = temp_word[-2:]
        temp_word = temp_word[:-2]
    elif temp_word.endswith("s'"):
        possessive = temp_word[-2:]
        temp_word = temp_word[:-2]

    # Detect trailing punctuation (after possessive removed)
    trailing_punct = ""
    for char in reversed(temp_word):
        if char in ".,!?;:'\"()[]{}":
            trailing_punct = char + trailing_punct
        else:
            break

    words_list[position] = corrected_value + possessive + trailing_punct
    return words_list


# ============================================================================
# STAGE 1 — RULE-BASED STREET NAME CORRECTION
# ============================================================================

def normalize_street_dictionary(streets: List[str]) -> List[str]:
    """
    Build a lookup list of street-name tokens (full names, sub-phrases,
    and individual words) with standard suffixes stripped.

    Filters out common words that should never match alone (French
    articles, etc.).
    """
    normalized = []
    seen: set = set()

    blacklist = {
        "de", "la", "du", "des", "le", "les", "rue", "port",
        "fort", "st", "saint",
    }

    street_suffixes = [
        "Street", "St.", "Avenue", "Ave.", "Boulevard", "Blvd.",
        "Road", "Rd.", "Drive", "Dr.", "Lane", "Ln.", "Court", "Ct.",
        "Circle", "Cir.", "Place", "Pl.", "Way", "Parkway", "Pkwy.",
        "Terrace", "Ter.", "Trail", "Alley",
    ]

    for street in streets:
        street_name = street
        for suffix in street_suffixes:
            if street.endswith(suffix):
                street_name = street[: -len(suffix)].strip()
                break

        if street_name not in seen:
            normalized.append(street_name)
            seen.add(street_name)

        words = street_name.split()
        if len(words) > 1:
            for word in words:
                if word.lower() not in blacklist and word not in seen:
                    normalized.append(word)
                    seen.add(word)

            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    phrase = " ".join(words[i:j])
                    first_word = words[i].lower()
                    if (
                        phrase not in seen
                        and phrase != street_name
                        and first_word not in blacklist
                    ):
                        normalized.append(phrase)
                        seen.add(phrase)

    return normalized


def fix_street_names_fuzzy(
    text: str,
    street_dictionary: List[str],
    english_words: set,
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Deterministic street-name correction via fuzzy matching with
    prefix-based score boosting.

    Returns (corrected_text, corrections, near_misses).
    """
    print("\n[correction] STAGE 1: RULE-BASED STREET CORRECTION (FUZZY)")

    normalized_streets = normalize_street_dictionary(street_dictionary)
    words = text.split()
    corrections = []
    near_misses = []

    street_indicators = [
        "street", "st.", "avenue", "ave.", "boulevard", "blvd.",
        "road", "rd.", "drive", "dr.", "lane", "ln.", "court", "ct.",
        "circle", "cir.", "place", "pl.", "way", "parkway", "pkwy.",
        "terrace", "ter.", "trail", "alley",
    ]

    street_dict_lower = {s.lower(): s for s in normalized_streets}
    corrected_positions: set = set()

    for i, word in enumerate(words):
        word_clean = word.strip(".,!?;:'\"()[]{}").lower()

        if word_clean in street_indicators and i > 0:
            for num_words in [3, 2, 1]:
                if i >= num_words:
                    potential_words = []
                    for j_idx in range(num_words):
                        ws = words[i - num_words + j_idx].strip(".,!?;:'\"()[]{}") 
                        if ws:
                            potential_words.append(ws)

                    if not potential_words:
                        continue

                    potential_street = " ".join(potential_words)
                    start_pos = i - num_words

                    if start_pos in corrected_positions:
                        break
                    if potential_words[0].replace(",", "").replace(".", "").isdigit():
                        continue
                    if any(pw.lower() in english_words for pw in potential_words):
                        continue
                    if potential_street.lower() in street_dict_lower:
                        break

                    # Fuzzy match with prefix-based score boosting
                    best_match = None
                    best_score = 0.0
                    potential_lower = potential_street.lower()

                    for correct_lower, correct_proper in street_dict_lower.items():
                        base_score = SequenceMatcher(
                            None, potential_lower, correct_lower
                        ).ratio()

                        # Prefix boost: common truncation typo pattern
                        if correct_lower.startswith(potential_lower):
                            score = min(base_score + 0.3, 1.0)
                        elif potential_lower.startswith(
                            correct_lower[: len(potential_lower) // 2]
                        ):
                            score = min(base_score + 0.1, 1.0)
                        else:
                            score = base_score

                        if score > best_score:
                            best_score = score
                            best_match = correct_proper

                    threshold = 0.75 if num_words == 1 else 0.65

                    # Reject very-short match candidates unless very high score
                    if best_match and len(best_match) <= 2 and best_score < 0.90:
                        continue

                    if best_score >= threshold:
                        corrections.append(
                            {
                                "original": potential_street,
                                "corrected": best_match,
                                "confidence": best_score,
                                "position": start_pos,
                                "num_words": num_words,
                                "context": " ".join(
                                    words[max(0, i - 6) : min(len(words), i + 3)]
                                ),
                            }
                        )
                        for k in range(num_words):
                            corrected_positions.add(start_pos + k)
                        print(
                            f"   WILL FIX: '{potential_street} {word}' -> "
                            f"'{best_match} {word}' ({best_score:.0%})"
                        )
                        break
                    elif best_score > 0.5:
                        near_misses.append(
                            {
                                "original": potential_street,
                                "best_candidate": best_match,
                                "score": best_score,
                                "threshold": threshold,
                                "position": start_pos,
                                "context": " ".join(
                                    words[max(0, i - 6) : min(len(words), i + 3)]
                                ),
                            }
                        )

    # Apply corrections in reverse position order to preserve indices
    corrected_text = text
    actual_corrections = []

    for correction in sorted(corrections, key=lambda x: -x["position"]):
        words_before = corrected_text.split()
        num_words = correction["num_words"]

        if correction["position"] + num_words <= len(words_before):
            corrected_parts = correction["corrected"].split()
            originals = [
                words_before[correction["position"] + j_idx]
                for j_idx in range(num_words)
                if correction["position"] + j_idx < len(words_before)
            ]
            words_after = words_before.copy()

            for j_idx in range(num_words):
                idx = correction["position"] + j_idx
                if idx < len(words_after) and j_idx < len(corrected_parts):
                    if j_idx == num_words - 1:
                        words_after = apply_correction_preserving_punctuation(
                            words_after, idx, corrected_parts[j_idx]
                        )
                    else:
                        words_after[idx] = corrected_parts[j_idx]

            changed = any(
                words_before[correction["position"] + j_idx]
                != words_after[correction["position"] + j_idx]
                for j_idx in range(num_words)
                if correction["position"] + j_idx < len(words_before)
            )

            if changed:
                actual_corrections.append(
                    {
                        "original": " ".join(originals),
                        "corrected": " ".join(
                            words_after[correction["position"] + j_idx]
                            for j_idx in range(num_words)
                            if correction["position"] + j_idx < len(words_after)
                        ),
                        "position": correction["position"],
                        "num_words": num_words,
                        "context": correction["context"],
                    }
                )
            corrected_text = " ".join(words_after)

    print(f"   Fixed {len(actual_corrections)} street name(s)")
    return corrected_text, actual_corrections, near_misses


# ============================================================================
# STAGE 2 — FUZZY-BASED NAME CORRECTION (NO LLM)
# ============================================================================

def fix_names_fuzzy(
    text: str, dicts: Dict
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Correct person names using only fuzzy matching against the names
    dictionary.  No LLM is used.

    Returns (corrected_text, corrections, near_misses).
    """
    print("[correction] STAGE 2: RULE-BASED NAME CORRECTION (FUZZY)")

    words = text.split()
    corrections = []
    near_misses = []

    title_words = [
        "mayor", "councilmember", "commissioner", "dr.", "mr.", "ms.",
        "mrs.", "madam", "miss", "professor", "senator", "representative",
        "director", "councilmembers", "member", "president", "councilwoman",
        "officer", "members", "councilor", "councilman", "vice",
    ]

    all_names = dicts["names"]
    english_words = dicts["english"]
    names_lower_set = {n.lower() for n in all_names}

    # Build frequency map (all case variants)
    name_frequencies: Counter = Counter()
    for w in words:
        clean = w.strip(".,!?;:'\"()[]{}") 
        if clean and len(clean) > 1:
            name_frequencies[clean] += 1

    for i, word in enumerate(words):
        if word.lower() not in title_words or i + 1 >= len(words):
            continue

        potential_name = words[i + 1].strip(".,!?;:'\"()[]{}") 

        if not potential_name:
            continue
        if potential_name.lower() in english_words:
            continue
        if potential_name.lower() in names_lower_set:
            continue

        close_matches = get_close_matches(
            potential_name, all_names, n=1, cutoff=0.70
        )
        if not close_matches:
            continue

        match = close_matches[0]
        if potential_name.lower() == match.lower():
            continue

        similarity = SequenceMatcher(
            None, potential_name.lower(), match.lower()
        ).ratio()

        # Confidence boosting based on document frequency
        confidence_boost = False

        correct_count = sum(
            name_frequencies.get(v, 0)
            for v in [match, match.lower(), match.upper(), match.capitalize()]
        )
        incorrect_count = sum(
            name_frequencies.get(v, 0)
            for v in [
                potential_name,
                potential_name.lower(),
                potential_name.upper(),
                potential_name.capitalize(),
            ]
        )

        if correct_count > incorrect_count and correct_count > 0:
            confidence_boost = True
        if incorrect_count <= 2:
            confidence_boost = True
        if correct_count >= 3:
            confidence_boost = True

        if similarity > 0.78 or (similarity > 0.70 and confidence_boost):
            corrections.append(
                {
                    "original": potential_name,
                    "corrected": match,
                    "confidence": similarity,
                    "position": i + 1,
                    "context": " ".join(
                        words[max(0, i - 3) : min(len(words), i + 8)]
                    ),
                }
            )
            print(
                f"   WILL FIX: '{potential_name}' -> '{match}' "
                f"({similarity:.0%}, after '{word}')"
            )
        elif similarity > 0.70:
            near_misses.append(
                {
                    "original": potential_name,
                    "best_candidate": match,
                    "score": similarity,
                    "threshold": 0.78 if not confidence_boost else 0.70,
                    "boosted": confidence_boost,
                    "position": i + 1,
                    "context": " ".join(
                        words[max(0, i - 3) : min(len(words), i + 8)]
                    ),
                }
            )

    # Apply corrections in reverse order
    corrected_text = text
    actual_corrections = []

    for correction in sorted(corrections, key=lambda x: -x["position"]):
        words_before = corrected_text.split()
        pos = correction["position"]

        if pos >= len(words_before):
            continue

        words_after = apply_correction_preserving_punctuation(
            words_before.copy(), pos, correction["corrected"]
        )

        if words_before[pos] != words_after[pos]:
            actual_corrections.append(
                {
                    "original": words_before[pos],
                    "corrected": words_after[pos],
                    "position": pos,
                    "context": correction["context"],
                }
            )
            print(
                f"   APPLIED: '{words_before[pos]}' -> '{words_after[pos]}'"
            )

        corrected_text = " ".join(words_after)

    print(f"   Fixed {len(actual_corrections)} name(s)")
    return corrected_text, actual_corrections, near_misses


# ============================================================================
# STAGE 0 — HARDCODED CORRECTIONS (known frequent Whisper errors)
# ============================================================================



# This gets populated at load time by load_hardcoded_corrections().
HARDCODED_CORRECTIONS: Dict = {}


def load_hardcoded_corrections(
    path: str = "hardcoded_corrections.json",
) -> Dict:
    """
    Load hardcoded corrections from a JSON file, merged with built-in
    defaults.  The JSON file is a simple object: {"misspelling": "Correct", ...}
    Keys are stored lowercase for case-insensitive matching.

    If the file doesn't exist, only the built-in defaults are used.
    """
    global HARDCODED_CORRECTIONS

    # Start with built-in defaults
    merged = {}


    # Layer on anything from the JSON file
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_corrections = json.load(f)
        # Normalize keys to lowercase
        for k, v in user_corrections.items():
            merged[k.lower()] = v
        print(f"[correction] Loaded {len(user_corrections)} hardcoded corrections from {path}")
    except FileNotFoundError:
        print(f"[correction] No {path} found — using {len(merged)} built-in defaults")
    except json.JSONDecodeError as e:
        print(f"[correction] WARNING: {path} has invalid JSON ({e}) — using defaults")

    HARDCODED_CORRECTIONS = merged
    print(f"[correction] Total hardcoded corrections: {len(HARDCODED_CORRECTIONS)}")
    return HARDCODED_CORRECTIONS


# Load on import
load_hardcoded_corrections()


def fix_hardcoded(text: str) -> Tuple[str, List[Dict]]:
    """
    Stage 0: Fix known, frequent transcription errors via exact
    case-insensitive matching.  Runs before fuzzy stages so that
    downstream stages don't waste effort on these.

    Preserves trailing punctuation and possessives.

    Returns (corrected_text, corrections).
    """
    print("[correction] STAGE 0: HARDCODED CORRECTIONS")

    words = text.split()
    corrections: List[Dict] = []

    for i, word in enumerate(words):
        # Strip punctuation/possessives to get the bare token
        bare = word.strip(".,!?;:'\"()[]{}") 
        # Also strip possessive for matching
        match_key = bare
        if match_key.endswith("'s") or match_key.endswith("\u2019s"):
            match_key = match_key[:-2]
        elif match_key.endswith("s'"):
            match_key = match_key[:-2]

        correct = HARDCODED_CORRECTIONS.get(match_key.lower())
        if correct and match_key.lower() != correct.lower():
            original_word = words[i]
            words = apply_correction_preserving_punctuation(
                words, i, correct
            )
            corrections.append(
                {
                    "original": original_word,
                    "corrected": words[i],
                    "position": i,
                    "context": " ".join(
                        words[max(0, i - 3) : min(len(words), i + 4)]
                    ),
                }
            )
            print(f"   FIXED: '{original_word}' -> '{words[i]}'")

    print(f"   Fixed {len(corrections)} hardcoded error(s)")
    return " ".join(words), corrections


# ============================================================================
# MAIN ENTRY POINT — THREE-STAGE FUZZY PIPELINE
# ============================================================================

def correct_transcript(text: str, dicts: Dict) -> Tuple[str, List[Dict]]:
    """
    Run the full three-stage correction pipeline on a piece of text.

    No LLM is called — all stages use deterministic matching.

    Stage 0: Hardcoded fixes for known frequent Whisper errors
    Stage 1: Fuzzy street name correction
    Stage 2: Fuzzy person name correction

    Args:
        text:  raw transcript text (e.g. concatenated retrieved chunks)
        dicts: dictionary bundle from load_dictionaries()

    Returns:
        (corrected_text, all_corrections)
        where each correction is a dict with keys:
            original, corrected, context, position, type
    """
    print("\n[correction] -- THREE-STAGE FUZZY CORRECTION PIPELINE (NO LLM) --")

    word_count = len(text.split())
    print(f"[correction] Input: {word_count} words")

    # Stage 0 — hardcoded known errors
    text_s0, hardcoded_corrs = fix_hardcoded(text)

    # Stage 1 — streets
    text_s1, street_corrs, _ = fix_street_names_fuzzy(
        text_s0, dicts["streets"], dicts["english"]
    )

    # Stage 2 — names
    text_s2, name_corrs, _ = fix_names_fuzzy(text_s1, dicts)

    # Merge with type labels
    all_corrections: List[Dict] = []
    for c in hardcoded_corrs:
        all_corrections.append({**c, "type": "HARDCODED"})
    for c in street_corrs:
        all_corrections.append({**c, "type": "STREET (Fuzzy)"})
    for c in name_corrs:
        all_corrections.append({**c, "type": "NAME (Fuzzy)"})
    all_corrections.sort(key=lambda x: x.get("position", 0))

    print(
        f"[correction] DONE — hardcoded: {len(hardcoded_corrs)}, "
        f"streets: {len(street_corrs)}, names: {len(name_corrs)}, "
        f"total: {len(all_corrections)}"
    )
    return text_s2, all_corrections