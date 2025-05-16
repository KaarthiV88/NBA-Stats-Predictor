import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog as pgl
from nba_api.stats.endpoints import teamdashboardbygeneralsplits
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import bet_calculations as bc


class PlayerStatOverUnderModel:
    def __init__(self):
        self.model = None
        self.scaler = None

    def prepare_features(self, player_id, opponent_abbr, category, season='2024-25', season_type='Regular Season'):
        # Get player stats
        season_avgs = bc.get_player_season_recent_averages(player_id, season, season_type)['season_averages']
        recent_avgs = bc.get_player_season_recent_averages(player_id, season, season_type)['recent_averages']
        h2h_avgs = bc.get_player_vs_opponent_averages(player_id, opponent_abbr, seasons=[season,]) or pd.Series(dtype='float64')
        
        # Get team stats
        opp_team_id = bc.get_team_id(opponent_abbr)
        opp_def_stats = bc.get_defensive_team_stats(opp_team_id, season, season_type).iloc[0]
        opp_off_stats = bc.get_offensive_team_stats(opp_team_id, season, season_type).iloc[0]

        # Combine features into one DataFrame
        features = pd.DataFrame({
            'season_pts': season_avgs.get('PTS', 0),
            'recent_pts': recent_avgs.get('PTS', 0),
            'h2h_pts': h2h_avgs.get('PTS', 0),
            'season_reb': season_avgs.get('REB', 0),
            'recent_reb': recent_avgs.get('REB', 0),
            'h2h_reb': h2h_avgs.get('REB', 0),
            'season_ast': season_avgs.get('AST', 0),
            'recent_ast': recent_avgs.get('AST', 0),
            'h2h_ast': h2h_avgs.get('AST', 0),
            'opp_def_rating': opp_def_stats.get('DEF_RATING', 0),
            'opp_off_rating': opp_off_stats.get('OFF_RATING', 0),
            'opp_pts_allowed': opp_def_stats.get('Opp_PTS', 0),
            'opp_reb_allowed': opp_def_stats.get('Opp_REB', 0),
            'opp_ast_allowed': opp_def_stats.get('Opp_AST', 0),
            # You can add more features relevant to the category here
        }, index=[0])

        # Select only relevant columns based on category
        if category.lower() == 'points':
            feature_cols = ['season_pts', 'recent_pts', 'h2h_pts', 'opp_def_rating', 'opp_pts_allowed']
        elif category.lower() == 'rebounds':
            feature_cols = ['season_reb', 'recent_reb', 'h2h_reb', 'opp_def_rating', 'opp_reb_allowed']
        elif category.lower() == 'assists':
            feature_cols = ['season_ast', 'recent_ast', 'h2h_ast', 'opp_def_rating', 'opp_ast_allowed']
        elif category.lower() == 'points+rebounds+assists':
            feature_cols = [
                'season_pts', 'recent_pts', 'h2h_pts', 
                'season_reb', 'recent_reb', 'h2h_reb',
                'season_ast', 'recent_ast', 'h2h_ast',
                'opp_def_rating', 'opp_pts_allowed', 'opp_reb_allowed', 'opp_ast_allowed'
            ]
        else:
            raise ValueError("Category not supported")

        return features[feature_cols]

    def train(self, training_data, training_labels):
        """
        Train the Random Forest model on historical data.
        training_data: DataFrame of features
        training_labels: 0 or 1 labels for under/over outcomes
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(training_data, training_labels)

    def predict_over_under_confidence(self, features):
        """
        Predict probability of Over or Under.
        Returns a dict: {'over': prob_over, 'under': prob_under}
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")

        prob = self.model.predict_proba(features)[0]  # [prob_under, prob_over]
        return {'under': prob[0], 'over': prob[1]}

    def evaluate_risk(self, prob_dict, threshold=0.8):
        """
        Evaluate if the bet confidence is strong enough.
        Returns string advice.
        """
        over_prob = prob_dict['over']
        under_prob = prob_dict['under']

        if over_prob >= threshold:
            return f"{over_prob*100:.1f}% confident bet on the OVER"
        elif under_prob >= threshold:
            return f"{under_prob*100:.1f}% confident bet on the UNDER"
        else:
            return f"Confidence too low (Over: {over_prob*100:.1f}%, Under: {under_prob*100:.1f}%), bet is risky."