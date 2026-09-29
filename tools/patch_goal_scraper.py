from pathlib import Path
import re
import sys

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scraper_repo/scrapers/unibet_event_goals_parser_v1.py")
s = target.read_text(encoding="utf-8")

labels_block = """BLOCK_LABEL_CANDIDATES = [
    "Nombre de Buts - Joueur - Match",
    "Nombre de Buts - Joueur",
    "NOMBRE DE BUTS - JOUEUR - MATCH",
    "NOMBRE DE BUTS - JOUEUR",
    "NOMBRE DE BUTS DU JOUEUR (PROLONGATIONS INCLUSES)",
    "NOMBRE DE BUTS DU JOUEUR",
    "BUTEUR (PROLONGATIONS INCLUSES)",
    "BUTEUR",
    "Buteur",
]"""
s2 = re.sub(r"BLOCK_LABEL_CANDIDATES\s*=\s*\[.*?\]\s*\n", labels_block + "\n", s, count=1, flags=re.S)
if s2 == s:
    raise RuntimeError("BLOCK_LABEL_CANDIDATES replacement failed")
s = s2

tab_block = """TAB_LABEL_CANDIDATES = [
    "Joueurs",
    "Buteurs",
    "Buteur",
    "Buts",
    "Points",
]"""
s2 = re.sub(r"TAB_LABEL_CANDIDATES\s*=\s*\[.*?\]\s*\n", tab_block + "\n", s, count=1, flags=re.S)
if s2 == s:
    raise RuntimeError("TAB_LABEL_CANDIDATES replacement failed")
s = s2

start = s.index("def select_first_matching_market_block(page, labels):")
end = s.index("\ndef click_all_see_more_in_block", start)
new_select = r'''def select_first_matching_market_block(page, labels):
    candidates = []

    for label in labels:
        locators = [
            page.get_by_text(label, exact=True),
            page.locator(f"text={label}"),
            page.get_by_text(label, exact=False),
        ]

        for loc in locators:
            try:
                count = safe_count(loc, 20)
            except Exception:
                count = 0
            for idx in range(min(count, 20)):
                try:
                    heading = loc.nth(idx)
                    if not heading.is_visible(timeout=1200):
                        continue

                    for level in range(1, 7):
                        try:
                            block = heading.locator("xpath=" + "/.." * level).first
                            text = safe_inner_text(block)
                            norm = normalize_for_match(text)
                            if not norm:
                                continue

                            lines = [x for x in text.splitlines() if norm_spaces(x)]
                            if "buteur" not in norm:
                                continue
                            if "2 buts ou plus" not in norm and len(lines) < 8:
                                continue

                            score = 0.0
                            if "2 buts ou plus" in norm:
                                score += 50
                            if "buteur" in norm:
                                score += 30
                            score += min(len(lines), 100) * 0.5
                            if "nombre de points du joueur" in norm:
                                score -= 100
                            if "buteur double chance" in norm:
                                score -= 100

                            candidates.append((score, -len(text), label, block))
                        except Exception:
                            continue
                except Exception:
                    continue

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, label, block = candidates[0]
    log(f"market block selected: {label}")
    return "BUTEUR", block
'''
s = s[:start] + new_select + s[end:]

start = s.index("def parse_goals_rows(lines, teams):")
end = s.index("\ndef validate_rows", start)
new_parse = r'''def parse_goals_rows(lines, teams):
    rows = []
    debug_players = []

    team_map = {}
    for team in teams:
        key = normalize_for_match(team)
        compact = re.sub(r"[^a-z0-9]+", "", key)
        tokens = [t for t in key.split() if len(t) >= 3]
        variants = {key, compact}
        if tokens:
            variants.add(tokens[-1])
        for variant in variants:
            if variant:
                team_map[variant] = team

    def resolve_team(line):
        key = normalize_for_match(line)
        compact = re.sub(r"[^a-z0-9]+", "", key)
        if key in team_map:
            return team_map[key]
        if compact in team_map:
            return team_map[compact]
        for variant, team in team_map.items():
            if len(variant) >= 5 and (variant in compact or compact in variant):
                return team
        return None

    i = 0
    current_team = None
    headers = {"buteur", "1 but", "1 but ou plus", "2 buts ou plus", "2 buts"}

    while i < len(lines):
        line = norm_spaces(lines[i])
        resolved = resolve_team(line)

        if resolved:
            current_team = resolved
            i += 1
            continue

        if not current_team:
            i += 1
            continue

        player = line
        if not is_valid_player_name(player, teams):
            i += 1
            continue

        j = i + 1
        odds = []
        while j < len(lines) and len(odds) < 4:
            token = norm_spaces(lines[j])
            key = normalize_for_match(token)

            if resolve_team(token):
                break
            if key in headers or "afficher plus" in key or "voir plus" in key or "voir moins" in key:
                break

            if is_decimal_odd(token) or token == "-":
                odds.append(token)
                j += 1
            else:
                break

        if odds:
            first = odds[0]
            debug_players.append({
                "team": current_team,
                "player_name_raw": player,
                "odds_count_seen": len(odds),
                "kept_outcome_label": "Buteur",
                "kept_odds_values": [first] if first != "-" else [],
                "parser_mode": "line_based_fuzzy_team",
            })
            if first != "-":
                rows.append({
                    "team": current_team,
                    "player_name_raw": player,
                    "outcome_label": "Buteur",
                    "odds_raw": first,
                })
            i = j
        else:
            i += 1

    return rows, debug_players
'''
s = s[:start] + new_parse + s[end:]

if "data-oai-goals-market-target" not in s:
    raise RuntimeError("Goal scraper patch verification failed")

target.write_text(s, encoding="utf-8")
print(f"patched {target}")
