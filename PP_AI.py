from predictive_model import AdvancedNBAPlayerPredictor
import bet_calculations as bc

def get_category_choice(choice):
    categories = {
        '1': 'POINTS',
        '2': 'REBOUNDS',
        '3': 'ASSISTS',
        '4': 'BLOCKS',
        '5': 'STEALS',
        '6': 'POINTS+REBOUNDS+ASSISTS',
        '7': 'REBOUNDS+ASSISTS',
        '8': 'POINTS+REBOUNDS',
        '9': 'POINTS+ASSISTS',
        '10': 'BLOCKS+STEALS'
    }
    return categories.get(choice, 'POINTS')

def main():
    try:
        player_name = input("Name of NBA Player: ").strip()
        print("What category is the bet:")
        print("1) Points\n2) Rebounds\n3) Assists\n4) Blocks\n5) Steals")
        print("6) Points+Rebounds+Assists\n7) Rebounds+Assists\n8) Points+Rebounds")
        print("9) Points+Assists\n10) Blocks+Steals")
        category_choice = input().strip()
        category = get_category_choice(category_choice)
        opponent_abbr = input("Enter the abbreviation of the opposing team (e.g., BOS): ").strip().upper()
        print("1) Regular Season\n2) Playoffs")
        season_type_choice = input().strip()
        season_type = 'Playoffs' if season_type_choice == '2' else 'Regular Season'
        betting_line = float(input("Enter the betting line: "))

        player_id = bc.get_player_id(player_name)
        predictor = AdvancedNBAPlayerPredictor()
        result = predictor.predict_over_under(
            player_id=player_id,
            category=category,
            opponent_abbr=opponent_abbr,
            season_type=season_type,
            betting_line=betting_line,
            category_type='offensive' if category in ['POINTS', 'REBOUNDS', 'ASSISTS', 'POINTS+REBOUNDS+ASSISTS', 'REBOUNDS+ASSISTS', 'POINTS+REBOUNDS', 'POINTS+ASSISTS'] else 'defensive',
            seasons=['2023-24', '2024-25']
        )

        print("\n" + result['message'])

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
