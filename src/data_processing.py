"""Data loading, cleaning, name standardization, and team-season merging."""

from pathlib import Path
import pandas as pd


TEAM_NAME_MAP = {
    "Boise State": "Boise St.",
    "Colorado State": "Colorado St.",
    "Florida State": "Florida St.",
    "Iowa State": "Iowa St.",
    "Loyola Maryland": "Loyola MD",
    "Miami": "Miami FL",
    "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.",
    "Missouri State": "Missouri St.",
    "Murray State": "Murray St.",
    "New Mexico State": "New Mexico St.",
    "North Carollina": "North Carolina",
    "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.",
    "San Diego State": "San Diego St.",
    "South Dakota State": "South Dakota St.",
    "Texas State": "Texas St.",
    "Utah State": "Utah St.",
    "Weber State": "Weber St.",
}


def standardize_columns(df):
    """Strip column names, uppercase them, and replace spaces with underscores."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
    return df


def clean_team_names(df, mapping=TEAM_NAME_MAP):
    """Strip team names and apply the shared team-name mapping."""
    df = df.copy()
    df["TEAM"] = df["TEAM"].str.strip().replace(mapping)
    return df


def load_raw_data(data_dir="../data/raw"):
    """Load the four raw team-stat datasets used in the project."""
    data_dir = Path(data_dir)

    ratings_538 = pd.read_csv(data_dir / "538 Ratings.csv")
    ap_polls = pd.read_csv(data_dir / "AP Poll Data.csv")
    kenpom = pd.read_csv(data_dir / "KenPom Barttorvik.csv")
    team_stats = pd.read_csv(data_dir / "cbb.csv")

    return ratings_538, ap_polls, kenpom, team_stats


def prepare_ap_poll(ap_polls):
    """Keep the final AP poll observation for each team-season through 2024."""
    ap = standardize_columns(ap_polls)
    ap = ap[ap["YEAR"] <= 2024].copy()
    ap = ap.sort_values("WEEK")
    ap = ap.groupby(["YEAR", "TEAM"], as_index=False).last()
    return ap[["YEAR", "TEAM", "AP_RANK"]]


def prepare_team_sources(ratings_538, ap_polls, kenpom, team_stats,
                         start_year=2016, end_year=2024):
    """Clean, align, and select the columns used from each team-stat source."""
    ratings_538 = standardize_columns(ratings_538)
    kenpom = standardize_columns(kenpom)
    team_stats = standardize_columns(team_stats)
    ap = prepare_ap_poll(ap_polls)

    sources = [ratings_538, ap, kenpom, team_stats]
    sources = [
        df[df["YEAR"].between(start_year, end_year)].copy()
        for df in sources
    ]
    ratings_538, ap, kenpom, team_stats = sources

    ratings_538 = clean_team_names(ratings_538)
    ap = clean_team_names(ap)
    kenpom = clean_team_names(kenpom)
    team_stats = clean_team_names(team_stats)

    # Year-specific corrections from the exploration notebook.
    ratings_538.loc[
        (ratings_538["TEAM"] == "Little Rock") & (ratings_538["YEAR"] == 2016),
        "TEAM"
    ] = "Arkansas Little Rock"

    ap.loc[
        (ap["TEAM"] == "Little Rock") & (ap["YEAR"] == 2016),
        "TEAM"
    ] = "Arkansas Little Rock"

    kenpom.loc[
        (kenpom["TEAM"] == "Little Rock") & (kenpom["YEAR"] == 2016),
        "TEAM"
    ] = "Arkansas Little Rock"

    ap.loc[
        (ap["TEAM"] == "Purdue Fort Wayne") & (ap["YEAR"] == 2017),
        "TEAM"
    ] = "Fort Wayne"

    ratings_538 = ratings_538[["TEAM", "YEAR", "POWER_RATING"]]
    ap = ap[["YEAR", "TEAM", "AP_RANK"]]
    kenpom = kenpom[
        ["TEAM", "YEAR", "KADJ_O", "KADJ_D", "BARTHAG", "EFG%", "EFG%D",
         "TOV%", "TOV%D", "OREB%", "DREB%", "FTR", "FTRD"]
    ]
    team_stats = team_stats[
        ["TEAM", "YEAR", "G", "W", "SEED", "2P_O", "2P_D",
         "3P_O", "3P_D", "ADJ_T", "WAB"]
    ]

    return ratings_538, ap, kenpom, team_stats


def build_team_seasons(ratings_538, ap, kenpom, team_stats):
    """Merge all cleaned sources into one row per team-season."""
    team_seasons = team_stats.copy()

    team_seasons = team_seasons.merge(kenpom, on=["TEAM", "YEAR"], how="left", validate="one_to_one")
    team_seasons = team_seasons.merge(ratings_538, on=["TEAM", "YEAR"], how="left", validate="one_to_one")
    team_seasons = team_seasons.merge(ap, on=["TEAM", "YEAR"], how="left", validate="one_to_one")

    return team_seasons


def create_team_seasons(data_dir="../data/raw",output_path="../data/processed/team_seasons.csv"):
    """Run the complete team-season preprocessing pipeline and save the result."""
    raw = load_raw_data(data_dir)
    ratings_538, ap, kenpom, team_stats = prepare_team_sources(*raw)
    team_seasons = build_team_seasons(ratings_538, ap, kenpom, team_stats)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_seasons.to_csv(output_path, index=False)

    return team_seasons
