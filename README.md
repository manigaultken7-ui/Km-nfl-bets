
# NFL Prop Edge V2

A probability-first NFL player-prop research dashboard.

## V2 additions
- Live player-prop ingestion from The Odds API
- Multiple sportsbook prices
- Paired Over/Under vig removal
- Consensus no-vig market probability
- Best available Over and Under price
- Full-slate scanning
- Time-aware model holdout
- Recent player form: 3/5/10-game windows
- Opportunity features: attempts, carries, targets, receptions
- Target share / carry share
- Opponent allowed production by position group
- Game total, spread, rest, home/away and roof context when available
- Historical injury/practice-report features from nflverse
- Automatic injury uncertainty guardrail for current props
- Monte Carlo outcome probability from empirical holdout residuals
- Estimated EV and ranking score
- CSV export

## Data sources
- nflverse via `nflreadpy`: weekly player stats, schedules, injury reports
- The Odds API: current NFL sportsbook markets and player props
- Open-Meteo module included for outdoor forecast context

## Install
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Live sportsbook setup
Create a The Odds API key and either:
1. paste it into the app sidebar, or
2. set the environment variable:

```bash
export THE_ODDS_API_KEY="YOUR_KEY"
```

Windows PowerShell:
```powershell
$env:THE_ODDS_API_KEY="YOUR_KEY"
```

## How the live ranking works
For each current prop:
1. Find paired Over and Under prices at each book.
2. Convert American odds to implied probabilities.
3. Normalize each pair to remove vig.
4. Average no-vig probabilities across books at the modal line.
5. Select the best available price for the chosen side.
6. Project the player's stat with a model trained only on earlier historical games.
7. Bootstrap historical holdout errors around the projection.
8. Estimate P(Over) / P(Under).
9. Compare that probability to the actual break-even probability of the best price.
10. Require minimum probability edge, minimum EV and minimum projection gap.
11. Withhold QUALIFIES when the player is currently Questionable, Doubtful or Out.

## Accuracy notes
This is intentionally conservative:
- No arbitrary injury penalty is hard-coded.
- No arbitrary weather yardage penalty is hard-coded.
- "Edge Score" is a ranking heuristic, not a predicted win percentage.
- A prop is not automatically good just because the projection clears the line.
- Holdout error matters.
- Price matters.
- Bookmaker vig matters.

## Best next upgrade (V3)
Build a snapshot database that automatically stores:
- timestamp
- bookmaker
- player
- prop market
- line
- Over/Under prices
- model projection
- model probability
- injury context
- weather
- spread/total

Then grade every result after the game and calculate:
- hit rate
- ROI
- closing-line value (CLV)
- Brier score
- calibration by probability bucket
- performance by market
- performance by edge bucket
- performance by sportsbook
- performance by day/time before kickoff

That historical audit is the step that tells you whether a model has a repeatable edge.
