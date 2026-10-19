"""Pure helpers for expert-cache budgeting."""


def expert_lru_budget(requested: int, *, has_static_pins: bool) -> int:
    """Honor a caller-reserved cache budget when static experts are pinned.

    Without pins, a token routes through 40 x 6 = 240 experts, so retain the
    historical 240-entry minimum needed for the active working set. Callers
    that reserve part of a total cache for pins pass the remaining LRU budget;
    forcing that remainder back to 240 silently doubled their requested cap.
    """
    return max(1 if has_static_pins else 240, int(requested))
