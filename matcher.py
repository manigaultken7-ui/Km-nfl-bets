
import re
import unicodedata
from rapidfuzz import process, fuzz

def normalize_name(name):
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())

def make_player_index(stats):
    latest = (
        stats.sort_values(["season", "week"])
        [["player_id", "player_display_name", "team", "position_group"]]
        .dropna(subset=["player_id", "player_display_name"])
        .drop_duplicates("player_id", keep="last")
        .copy()
    )
    latest["norm"] = latest["player_display_name"].map(normalize_name)
    return latest

def match_player(name, player_index, team=None, threshold=88):
    norm = normalize_name(name)
    candidates = player_index
    if team:
        team_subset = player_index[player_index["team"].astype(str) == str(team)]
        if not team_subset.empty:
            candidates = team_subset

    exact = candidates[candidates["norm"] == norm]
    if not exact.empty:
        return exact.iloc[0], 100.0

    choices = candidates["norm"].tolist()
    if not choices:
        return None, 0.0
    hit = process.extractOne(norm, choices, scorer=fuzz.ratio)
    if not hit or hit[1] < threshold:
        return None, float(hit[1]) if hit else 0.0
    row = candidates[candidates["norm"] == hit[0]].iloc[0]
    return row, float(hit[1])
