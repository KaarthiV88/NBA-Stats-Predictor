import nba_api
import pandas as pd
import tabulate
import bet_calculations as bc
from nba_api.stats.endpoints import playergamelog as pgl


playerName = str(input("Name of NBA Player: "))
bet_category = int(input("""What category is the bet:\n1) Points\n2) Rebounds\n3) Assists\n4) Points+Rebounds+Assists 
5) Points+Rebounds\n6) Points+Assists\n7) Assists+Rebounds\n"""))

opposingTeam = str(input("Enter the abbreviation of the opposing team: "))

dict_category = {
    1: "Points",
    2: "Rebounds",
    3: "Assists",
    4: "Points+Rebounds+Assists",
    5: "Points+Rebounds",
    6: "Points+Assists",
    7: "Assists+Rebounds"
}

print("Your bet is on the", playerName, dict_category[bet_category], "\n\n")

pgl.PlayerGameLog(player_id=bc.get_player_id(playerName), season='2024-25', season_type_all_star='Regular Season')
