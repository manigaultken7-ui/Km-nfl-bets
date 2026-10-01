
SPORT_KEY = "americanfootball_nfl"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

MARKETS = {
    "Passing Yards": {
        "target": "passing_yards",
        "api_key": "player_pass_yds",
        "positions": ["QB"],
        "opportunity": ["attempts", "completions"],
        "usage": [],
    },
    "Pass Attempts": {
        "target": "attempts",
        "api_key": "player_pass_attempts",
        "positions": ["QB"],
        "opportunity": ["passing_yards", "completions"],
        "usage": [],
    },
    "Pass Completions": {
        "target": "completions",
        "api_key": "player_pass_completions",
        "positions": ["QB"],
        "opportunity": ["attempts", "passing_yards"],
        "usage": [],
    },
    "Passing TDs": {
        "target": "passing_tds",
        "api_key": "player_pass_tds",
        "positions": ["QB"],
        "opportunity": ["attempts", "passing_yards"],
        "usage": [],
        "count_market": True,
    },
    "Rushing Yards": {
        "target": "rushing_yards",
        "api_key": "player_rush_yds",
        "positions": ["RB", "QB", "WR"],
        "opportunity": ["carries"],
        "usage": ["carry_share"],
    },
    "Rush Attempts": {
        "target": "carries",
        "api_key": "player_rush_attempts",
        "positions": ["RB", "QB", "WR"],
        "opportunity": ["rushing_yards"],
        "usage": ["carry_share"],
    },
    "Rushing TDs": {
        "target": "rushing_tds",
        "api_key": "player_rush_tds",
        "positions": ["RB", "QB", "WR"],
        "opportunity": ["carries", "rushing_yards"],
        "usage": ["carry_share"],
        "count_market": True,
    },
    "Receiving Yards": {
        "target": "receiving_yards",
        "api_key": "player_reception_yds",
        "positions": ["WR", "TE", "RB"],
        "opportunity": ["targets", "receptions"],
        "usage": ["target_share"],
    },
    "Receptions": {
        "target": "receptions",
        "api_key": "player_receptions",
        "positions": ["WR", "TE", "RB"],
        "opportunity": ["targets", "receiving_yards"],
        "usage": ["target_share"],
    },
    "Receiving TDs": {
        "target": "receiving_tds",
        "api_key": "player_reception_tds",
        "positions": ["WR", "TE", "RB"],
        "opportunity": ["targets", "receiving_yards"],
        "usage": ["target_share"],
        "count_market": True,
    },
}

DEFAULT_MARKETS = [
    "Passing Yards",
    "Rushing Yards",
    "Receiving Yards",
    "Receptions",
]

# Approximate home-stadium coordinates. Weather is skipped for neutral/international games.
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7553, -84.4006),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LV": (36.0908, -115.1830), "LAC": (33.9535, -118.3392),
    "LA": (33.9535, -118.3392), "MIA": (25.9580, -80.2389),
    "MIN": (44.9736, -93.2575), "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9078, -76.8645),
}
