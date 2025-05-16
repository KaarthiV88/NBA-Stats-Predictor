import nba_api
import pandas
import tabulate

nba_player = str(input("Name of NBA Player: "))
bet_category = int(input("""What category is the bet:\n1) Points\n2) Rebounds\n3) Assists\n4) Points+Rebounds+Assists 
5) Points+Rebounds\n6) Points+Assists\n7) Assists+Rebounds\n"""))

dict_category = {
    1: "Points",
    2: "Rebounds",
    3: "Assists",
    4: "Points+Rebounds+Assists",
    5: "Points+Rebounds",
    6: "Points+Assists",
    7: "Assists+Rebounds"
}

print("Your bet is on the", nba_player, dict_category[bet_category])