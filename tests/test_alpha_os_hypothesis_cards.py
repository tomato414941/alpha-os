from __future__ import annotations

import re
from pathlib import Path


HYPOTHESIS_UID_RE = re.compile(r"^Hypothesis UID: `(?P<uid>hyp_[0-9a-f]{8})`$")


def _hypothesis_cards() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    hypotheses = root / "experiments" / "hypotheses"
    cards = [
        *hypotheses.glob("*.md"),
        *hypotheses.glob("*/README.md"),
    ]
    return tuple(sorted(cards))


def _hypothesis_uids(path: Path) -> tuple[str, ...]:
    uids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HYPOTHESIS_UID_RE.match(line)
        if match is not None:
            uids.append(match.group("uid"))
    return tuple(uids)


def test_hypothesis_cards_have_exactly_one_uid():
    cards = _hypothesis_cards()
    assert cards

    for card in cards:
        assert len(_hypothesis_uids(card)) == 1, card


def test_hypothesis_uids_are_unique():
    uid_to_cards: dict[str, list[Path]] = {}
    for card in _hypothesis_cards():
        for uid in _hypothesis_uids(card):
            uid_to_cards.setdefault(uid, []).append(card)

    duplicates = {
        uid: cards for uid, cards in uid_to_cards.items() if len(cards) > 1
    }
    assert not duplicates
