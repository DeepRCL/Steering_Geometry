import json
import re


def _extract_jsons(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    results = []
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        results.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    pos = i + 1
                    break
        else:
            break
    return results


def parse_perturbation_output(raw: str, keys: list[str]) -> dict[str, str | None]:
    """Extract multiple keys from a perturbation JSON output string."""
    if "</think>" in raw:
        _, after = raw.split("</think>", 1)
    else:
        after = raw

    for jsons in (_extract_jsons(after), _extract_jsons(raw)):
        if jsons:
            obj = jsons[0]
            return {k: obj.get(k) for k in keys}

    return {k: None for k in keys}
