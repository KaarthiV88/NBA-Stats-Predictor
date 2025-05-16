import nba_api
import pandas as pd
import tabulate
import bet_calculations as bc
from nba_api.stats.endpoints import playergamelog as pgl


playerName = str(input("Name of NBA Player: "))
#bet_category = int(input("""What category is the bet:\n1) Points\n2) Rebounds\n3) Assists\n4) Points+Rebounds+Assists 
#5) Points+Rebounds\n6) Points+Assists\n7) Assists+Rebounds\n"""))

opposingTeam = str(input("Enter the abbreviation of the opposing team: "))
season_type_input = int(input("1) Regular Season\n2) Playoffs\n"))

dict_category = {
    1: "Points",
    2: "Rebounds",
    3: "Assists",
    4: "Points+Rebounds+Assists",
    5: "Points+Rebounds",
    6: "Points+Assists",
    7: "Assists+Rebounds"
}

dict_seasonType = {
    1: "Regular Season",
    2: "Playoffs"
}

opposingTeam_ABBR = bc.get_team_id(opposingTeam)

#print("Your bet is on the", playerName, dict_category[bet_category], "\n\n")

player_h2h_stats = bc.get_head_to_head_stats(bc.get_player_id(playerName), opposingTeam, seasons=["2023-24", "2024-25"], season_type=dict_seasonType[season_type_input])

print(pgl.PlayerGameLog(player_id=bc.get_player_id(playerName), season='2024-25', season_type_all_star=dict_seasonType[season_type_input]))
print(player_h2h_stats[['SEASON', 'GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']])

oppDef = bc.get_defensive_team_stats(opposingTeam_ABBR)
oppOff = bc.get_offensive_team_stats(opposingTeam_ABBR)

print()
print(oppDef[['GP', 'PTS', 'REB', 'AST', 'FG_PCT', 'DEF_RATING']])
print()
print(oppOff)
