import pandas as pd
import bet_calculations as bc
from predictive_model import AdvancedNBAPlayerPredictor

# User inputs
player_name = input("Name of NBA Player: ")
bet_category = int(input("What category is the bet:\n1) Points\n2) Rebounds\n3) Assists\n4) Blocks\n5) Steals\n6) Points+Rebounds+Assists\n7) Rebounds+Assists\n8) Points+Rebounds\n9) Points+Assists\n10) Blocks+Steals\n"))
opposing_team = input("Enter the abbreviation of the opposing team (e.g., BOS): ")
season_type_input = int(input("1) Regular Season\n2) Playoffs\n"))
betting_line = float(input("Enter the betting line: "))

# Betting category mapping
dict_category = {
    1: "Points",
    2: "Rebounds",
    3: "Assists",
    4: "Blocks",
    5: "Steals",
    6: "Points+Rebounds+Assists",
    7: "Rebounds+Assists",
    8: "Points+Rebounds",
    9: "Points+Assists",
    10: "Blocks+Steals"
}

# Determine category type
offensive_categories = ["Points", "Rebounds", "Assists", "Points+Rebounds+Assists", "Rebounds+Assists", "Points+Rebounds", "Points+Assists"]
defensive_categories = ["Blocks", "Steals", "Blocks+Steals"]
category = dict_category[bet_category]
category_type = "offensive" if category in offensive_categories else "defensive"

# Stat filters
offensive_stats = ["PTS", "REB", "AST"]
defensive_stats = ["BLK", "STL"]

try:
    # Get player and team IDs
    player_id = bc.get_player_id(player_name)
    opponent_abbr = opposing_team.upper()
    season_type = "Regular Season" if season_type_input == 1 else "Playoffs"

    # Fetch player head-to-head stats
    player_h2h_stats = bc.get_head_to_head_stats(player_id, opponent_abbr, seasons=["2023-24", "2024-25"], season_type=season_type)
    if not player_h2h_stats.empty:
        print(f"\nPlayer {'Offensive' if category_type == 'offensive' else 'Defensive'} Head-to-Head Stats vs {opponent_abbr}:")
        relevant_stats = offensive_stats if category_type == "offensive" else defensive_stats
        display_cols = ["SEASON", "GAME_DATE", "MATCHUP"] + [col for col in relevant_stats if col in player_h2h_stats.columns]
        print(player_h2h_stats[display_cols])

    # Fetch opposing team stats
    opp_team_id = bc.get_team_id(opponent_abbr)
    measure_type = "Defense" if category_type == "offensive" else "Base"
    opp_stats = bc.get_team_stats(opp_team_id, season="2023-24", season_type=season_type, measure_type=measure_type)
    if not opp_stats.empty:
        print(f"\nOpposing Team ({opponent_abbr}) {'Defensive' if category_type == 'offensive' else 'Offensive'} Stats:")
        relevant_opp_stats = ["OPP_PTS", "OPP_REB", "OPP_AST", "FG_PCT", "DEF_RATING"] if category_type == "offensive" else ["PTS", "REB", "AST", "FG_PCT"]
        available_cols = [col for col in ["TEAM_NAME", "GP"] + relevant_opp_stats if col in opp_stats.columns]
        print(opp_stats[available_cols])

    # Fetch player averages
    averages = bc.get_player_season_recent_averages(player_id, season="2023-24", season_type=season_type)
    if not averages["season_averages"].empty:
        print(f"\nPlayer {'Offensive' if category_type == 'offensive' else 'Defensive'} Season Averages:")
        relevant_avg_stats = offensive_stats if category_type == "offensive" else defensive_stats
        print(averages["season_averages"][relevant_avg_stats])
    if not averages["recent_averages"].empty:
        print(f"\nPlayer {'Offensive' if category_type == 'offensive' else 'Defensive'} Last 10 Games Averages:")
        print(averages["recent_averages"][relevant_avg_stats])

    # Make prediction
    predictor = AdvancedNBAPlayerPredictor()
    result = predictor.predict_over_under(
        player_id=player_id,
        category=category,
        opponent_abbr=opponent_abbr,
        season_type=season_type,
        betting_line=betting_line,
        category_type=category_type
    )
    print("\nBet Prediction:")
    print(result["message"])

except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
