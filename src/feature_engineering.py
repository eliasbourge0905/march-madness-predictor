"""Create tournament matchup features from historical tournament results."""

from pathlib import Path
import numpy as np
import pandas as pd


TOURNAMENT_NAME_MAP = {
    "ETSU": "East Tennessee St.",
    "FGCU": "Florida Gulf Coast",
    "MTSU": "Middle Tennessee",
    "WKU": "Western Kentucky",
    "NC State": "North Carolina St.",
    "WI Green Bay": "Green Bay",
    "Abilene Chr": "Abilene Christian",
    "Appalachian St": "Appalachian St.",
    "Arizona St": "Arizona St.",
    "Ark Little Rock": "Arkansas Little Rock",
    "Boise St": "Boise St.",
    "CS Bakersfield": "Cal St. Bakersfield",
    "CS Fullerton": "Cal St. Fullerton",
    "Cleveland St": "Cleveland St.",
    "Col Charleston": "College of Charleston",
    "Colorado St": "Colorado St.",
    "E Washington": "Eastern Washington",
    "F Dickinson": "Fairleigh Dickinson",
    "FL Atlantic": "Florida Atlantic",
    "Florida St": "Florida St.",
    "Fresno St": "Fresno St.",
    "Georgia St": "Georgia St.",
    "Grambling": "Grambling St.",
    "Iowa St": "Iowa St.",
    "Jacksonville St": "Jacksonville St.",
    "Kansas St": "Kansas St.",
    "Kennesaw": "Kennesaw St.",
    "Kent": "Kent St.",
    "Long Beach St": "Long Beach St.",
    "Louisiana": "Louisiana Lafayette",
    "Loyola-Chicago": "Loyola Chicago",
    "McNeese St": "McNeese St.",
    "Michigan St": "Michigan St.",
    "Mississippi St": "Mississippi St.",
    "Montana St": "Montana St.",
    "Morehead St": "Morehead St.",
    "Mt St Mary's": "Mount St. Mary's",
    "Murray St": "Murray St.",
    "N Dakota St": "North Dakota St.",
    "N Kentucky": "Northern Kentucky",
    "NC Central": "North Carolina Central",
    "New Mexico St": "New Mexico St.",
    "Norfolk St": "Norfolk St.",
    "Ohio St": "Ohio St.",
    "Oklahoma St": "Oklahoma St.",
    "Oregon St": "Oregon St.",
    "Penn St": "Penn St.",
    "Prairie View": "Prairie View A&M",
    "S Dakota St": "South Dakota St.",
    "SE Missouri St": "Southeast Missouri St.",
    "SF Austin": "Stephen F. Austin",
    "San Diego St": "San Diego St.",
    "Southern Univ": "Southern",
    "St Bonaventure": "St. Bonaventure",
    "St John's": "St. John's",
    "St Joseph's PA": "Saint Joseph's",
    "St Louis": "Saint Louis",
    "St Mary's CA": "Saint Mary's",
    "St Peter's": "Saint Peter's",
    "TAM C. Christi": "Texas A&M Corpus Chris",
    "TX Southern": "Texas Southern",
    "Utah St": "Utah St.",
    "Washington St": "Washington St.",
    "Weber St": "Weber St.",
    "Wichita St": "Wichita St.",
    "Wright St": "Wright St.",
}

TOURNAMENT_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]

DIFF_STATS = [
    "SEED", "WIN_PCT", "2P_O", "2P_D", "3P_O", "3P_D", "ADJ_T", "WAB",
    "KADJ_O", "KADJ_D", "BARTHAG", "EFG%", "EFG%D", "TOV%", "TOV%D",
    "OREB%", "DREB%", "FTR", "FTRD", "POWER_RATING", "AP_RANK"
]


def load_tournament_data(data_dir="../data/raw"):
    """Load tournament results and the TeamID-to-name lookup."""
    data_dir = Path(data_dir)
    results = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")
    teams = pd.read_csv(data_dir / "MTeams.csv")
    return results, teams


def prepare_tournament_results(tourney_results, teams,
                               years=TOURNAMENT_YEARS):
    """Filter tournament seasons and attach standardized team names."""
    results = tourney_results[tourney_results["Season"].isin(years)].copy()

    team_lookup = dict(zip(teams["TeamID"], teams["TeamName"]))
    results["WTeam"] = results["WTeamID"].map(team_lookup)
    results["LTeam"] = results["LTeamID"].map(team_lookup)

    results["WTeam"] = results["WTeam"].replace(TOURNAMENT_NAME_MAP)
    results["LTeam"] = results["LTeam"].replace(TOURNAMENT_NAME_MAP)

    return results


def merge_team_stats(results, team_seasons):
    """Attach season statistics for both the historical winner and loser."""
    games = results[
        ["Season", "DayNum", "WTeam", "LTeam", "WScore", "LScore"]
    ].copy()
    games = games.rename(columns={"Season": "YEAR"})

    winner_stats = team_seasons.add_prefix("W_")
    loser_stats = team_seasons.add_prefix("L_")

    games = games.merge(
        winner_stats,
        left_on=["WTeam", "YEAR"],
        right_on=["W_TEAM", "W_YEAR"],
        how="left",
        validate="many_to_one",
    )

    games = games.merge(
        loser_stats,
        left_on=["LTeam", "YEAR"],
        right_on=["L_TEAM", "L_YEAR"],
        how="left",
        validate="many_to_one",
    )

    return games


def assign_team_a_b(games, random_state=42):
    """Randomly assign the historical winner to Team A or B to avoid label bias."""
    games = games.copy()
    rng = np.random.default_rng(random_state)
    winner_is_a = rng.integers(0, 2, size=len(games)).astype(bool)

    games["TEAM_A"] = np.where(winner_is_a, games["WTeam"], games["LTeam"])
    games["TEAM_B"] = np.where(winner_is_a, games["LTeam"], games["WTeam"])
    games["TEAM_A_WIN"] = winner_is_a.astype(int)

    stat_cols = [
        col for col in games.columns
        if col.startswith("W_") and col not in ["W_TEAM", "W_YEAR"]
    ]

    # Convert W_STAT/L_STAT pairs into A_STAT/B_STAT pairs.
    for winner_col in stat_cols:
        stat = winner_col[2:]
        loser_col = f"L_{stat}"
        if loser_col not in games.columns:
            continue

        games[f"A_{stat}"] = np.where(
            winner_is_a, games[winner_col], games[loser_col]
        )
        games[f"B_{stat}"] = np.where(
            winner_is_a, games[loser_col], games[winner_col]
        )

    return games


def create_difference_features(games):
    """Create Team A minus Team B matchup features."""
    games = games.copy()

    games["A_WIN_PCT"] = games["A_W"] / games["A_G"]
    games["B_WIN_PCT"] = games["B_W"] / games["B_G"]

    for stat in DIFF_STATS:
        games[f"{stat}_DIFF"] = games[f"A_{stat}"] - games[f"B_{stat}"]

    diff_columns = [f"{stat}_DIFF" for stat in DIFF_STATS]

    # Excluded in the notebook:
    # AP_RANK_DIFF has substantial missingness.
    # WIN_PCT_DIFF may contain postseason information and create leakage.
    model_features = [
        col for col in diff_columns
        if col not in ["AP_RANK_DIFF", "WIN_PCT_DIFF"]
    ]

    return games, model_features


def build_tournament_matchups(team_seasons, tourney_results, teams,
                              random_state=42):
    """Build the final model-ready tournament matchup dataset."""
    results = prepare_tournament_results(tourney_results, teams)
    games = merge_team_stats(results, team_seasons)
    games = assign_team_a_b(games, random_state=random_state)
    games, model_features = create_difference_features(games)

    final_columns = [
        "YEAR", "DayNum", "TEAM_A", "TEAM_B", "TEAM_A_WIN"
    ] + model_features

    return games[final_columns].copy()


def create_tournament_matchups(
    team_seasons_path="../data/processed/team_seasons.csv",
    data_dir="../data/raw",
    output_path="../data/final/tournament_matchups.csv",
    random_state=42,
):
    """Run the full feature-engineering pipeline and save the final dataset."""
    team_seasons = pd.read_csv(team_seasons_path)
    results, teams = load_tournament_data(data_dir)

    matchups = build_tournament_matchups(
        team_seasons, results, teams, random_state=random_state
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(output_path, index=False)

    return matchups
