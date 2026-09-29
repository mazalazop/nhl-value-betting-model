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
    result = page.evaluate(
        """
        (cfg) => {
          const normalize = (v) => String(v || '')
            .toLowerCase()
            .replaceAll(String.fromCharCode(10), ' ')
            .replaceAll(String.fromCharCode(13), ' ')
            .replaceAll(String.fromCharCode(9), ' ')
            .replaceAll('  ', ' ')
            .trim();

          const labels = (cfg.labels || []).map(normalize);
          const all = Array.from(document.querySelectorAll('div,section,article,li'));

          for (const el of document.querySelectorAll('[data-oai-goals-market-target]')) {
            el.removeAttribute('data-oai-goals-market-target');
          }

          const candidates = [];
          for (const el of all) {
            const raw = el.innerText || '';
            const text = normalize(raw);
            if (!text) continue;

            const goalSignal =
              text.includes('buteur') ||
              text.includes('nombre de buts') ||
              text.includes('2 buts ou plus') ||
              text.includes('1 but ou plus');

            if (!goalSignal) continue;

            const lines = raw.split(String.fromCharCode(10)).map(x => x.trim()).filter(Boolean).length;
            const digitCount = Array.from(text).filter(c => c >= '0' && c <= '9').length;
            const more = text.includes('afficher plus') || text.includes('voir plus');
            const labelHit = labels.some(x => text.startsWith(x) || text.includes(x));

            let score = 0;
            if (labelHit) score += 250;
            if (text.startsWith('nombre de buts')) score += 220;
            if (text.startsWith('buteur')) score += 180;
            if (text.includes('2 buts ou plus')) score += 90;
            if (text.includes('1 but ou plus')) score += 60;
            if (digitCount >= 6) score += 40;
            if (lines >= 5 && lines <= 180) score += 25;
            if (more) score += 15;
            if (text.includes('nombre de points') && !text.includes('buteur') && !text.includes('nombre de buts')) score -= 300;
            if (text.includes('nombre de passes decisives') && !text.includes('buteur') && !text.includes('nombre de buts')) score -= 300;
            if (text.length > 10000) score -= 300;
            if (lines > 300) score -= 200;

            candidates.push({el, score, text_length:text.length, line_count:lines});
          }

          candidates.sort((a,b) => b.score-a.score || a.text_length-b.text_length || a.line_count-b.line_count);

          if (!candidates.length) return {found:false};

          candidates[0].el.setAttribute('data-oai-goals-market-target', '1');
          return {found:true, score:candidates[0].score, text_length:candidates[0].text_length};
        }
        """,
        {"labels": labels},
    )

    if not result.get("found"):
        return None, None

    return "BUTEUR", page.locator('[data-oai-goals-market-target="1"]').first
'''
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

if "Nombre de Buts - Joueur" not in s or "data-oai-goals-market-target" not in s:
    raise RuntimeError("Goal scraper patch verification failed")

target.write_text(s, encoding="utf-8")
print(f"patched {target}")
