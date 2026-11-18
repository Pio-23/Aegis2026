"""Review Gemma navigation decisions without controlling AEGIS."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CHOICES = {
    "g": "GEMMA_BETTER",
    "l": "LUNA_BETTER",
    "b": "BOTH_ACCEPTABLE",
    "n": "NEITHER_ACCEPTABLE",
    "u": "INSUFFICIENT_EVIDENCE",
}

def sample_id(row):
    data = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def load_lines(path):
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            print("Skipped malformed JSON record.")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trip", type=Path)
    parser.add_argument("--show-only", action="store_true")
    args = parser.parse_args()

    source = args.trip / "gemma_shadow_decisions.jsonl"
    output = args.trip / "gemma_coaching_reviews.jsonl"

    records = load_lines(source)
    if not records:
        print("No Gemma decisions found:", source)
        return

    reviewed = {
        r.get("sample_id") for r in load_lines(output)
    }

    print("Gemma decisions:", len(records))
    print("Already reviewed:", len(reviewed))

    for index, record in enumerate(records, 1):
        identifier = sample_id(record)

        if identifier in reviewed:
            continue

        if record.get("status") != "completed":
            continue

        gemma = (
            record.get("gemma_action")
            or record.get("gemma_decision")
            or {}
        )
        luna = record.get("luna_action") or {}

        if isinstance(luna, dict):
            name = luna.get("tool") or luna.get("action")
        else:
            name = None

        paired = name in (
            "move_rover",
            "no_op",
            "scan_environment",
            "capture_camera",
        )

        print("\n========== DECISION", index, "==========")
        print("Luna:", json.dumps(luna, default=str)[:500])
        print("Gemma:", json.dumps(gemma, default=str)[:900])
        print("Reason:", gemma.get("reason", "unknown"))
        context = record.get(
            "navigation_context",
            record.get("navigation_state")
        )
        print(
            "Navigation context:",
            json.dumps(context, default=str)[:650]
        )

        if not paired:
            print("\nOFFLINE REPLAY: No matching Luna decision.")
            print("[e] Record your observation")
            print("[s] Skip")
        else:
            print("\n[g] Gemma made the better decision")
            print("[l] Luna made the better decision")
            print("[b] Both were acceptable")
            print("[n] Neither was acceptable")
            print("[u] Not enough evidence")
            print("[s] Skip")

        if args.show_only:
            continue

        choice = input("Your choice: ").strip().lower()

        if not paired:
            if choice != "e":
                continue
            verdict = "OFFLINE_OBSERVATION_ONLY"
        else:
            if choice not in CHOICES:
                continue
            verdict = CHOICES[choice]

        note = input(
            "What happened, or what should Gemma learn? "
        ).strip()

        expected = input(
            "Preferred action "
            "(MOVE/TURN/SCAN/CAMERA/NO_OP/UNKNOWN): "
        ).strip().upper()

        if expected not in {
            "MOVE", "TURN", "SCAN", "CAMERA", "NO_OP"
        }:
            expected = "UNKNOWN"

        review = {
            "sample_id": identifier,
            "reviewed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "verdict": verdict,
            "paired_luna_decision": paired,
            "preferred_action": expected,
            "teaching_note": note,
            "original_decision": record,
        }

        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(review, default=str) + "\n")

        print("Review saved:", output)
        reviewed.add(identifier)

    print("\nReview complete.")
    print("Gemma model weights were not modified.")

if __name__ == "__main__":
    main()
