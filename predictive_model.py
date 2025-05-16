import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
from nba_api.stats.endpoints import playergamelog
import bet_calculations as bc

class AdvancedNBAPlayerPredictor:
    def __init__(self, n_components=0.95):
        self.regressor = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.stat_categories = ['PTS', 'REB', 'AST', 'BLK']
        self.feature_cols = None

    def prepare_data(self, player_id, seasons=['2023-24', '2024-25'], season_type='Regular Season'):
        game_logs = []
        for season in seasons:
            try:
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star=season_type)
                df = gamelog.get_data_frames()[0]
                df['SEASON'] = season
                game_logs.append(df)
            except:
                continue
        if not game_logs:
            raise ValueError("No game logs found for the player")
        all_games = pd.concat(game_logs).reset_index(drop=True)
        all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE']).dt.date
        current_date = datetime.now().date()
        all_games['days_ago'] = (current_date - all_games['GAME_DATE']).apply(lambda x: x.days)
        all_games['OPPONENT'] = all_games['MATCHUP'].apply(lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1])
        return all_games

    def create_features(self, all_games, opponent_abbr, season_type):
        # Recency weighting
        lambda_ = -np.log(0.5) / 30
        all_games['recency_weight'] = np.exp(-lambda_ * all_games['days_ago'])

        # Opponent strength for each stat
        opp_team_id = bc.get_team_id(opponent_abbr)
        opp_def_stats = bc.get_defensive_team_stats(opp_team_id, all_games['SEASON'].iloc[-1], season_type).iloc[0]
        league_avgs = bc.get_league_defensive_averages(all_games['SEASON'].iloc[-1], season_type)
        opponent_strengths = {}
        for stat in self.stat_categories:
            allowed_stat = f'Opp_{stat}'
            opp_allowed = opp_def_stats[allowed_stat]
            league_allowed = league_avgs[allowed_stat]
            opponent_strengths[stat] = opp_allowed / league_allowed if league_allowed != 0 else 1

        # Enhanced feature set
        all_games['same_opponent'] = (all_games['OPPONENT'] == opponent_abbr).astype(int)
        all_games['is_playoff'] = (season_type == 'Playoffs').astype(int)
        for stat in self.stat_categories:
            all_games[f'weighted_{stat}'] = all_games['recency_weight'] * all_games[stat]
            all_games[f'opp_strength_{stat}'] = opponent_strengths[stat]
            all_games[f'opp_interaction_{stat}'] = all_games['same_opponent'] * opponent_strengths[stat]
            all_games[f'recent_form_{stat}'] = all_games[stat].rolling(window=5, min_periods=1).mean()

        self.feature_cols = ['recency_weight', 'same_opponent', 'is_playoff'] + \
                            [f'weighted_{stat}' for stat in self.stat_categories] + \
                            [f'opp_strength_{stat}' for stat in self.stat_categories] + \
                            [f'opp_interaction_{stat}' for stat in self.stat_categories] + \
                            [f'recent_form_{stat}' for stat in self.stat_categories]

        X = all_games[self.feature_cols]
        y = all_games[self.stat_categories]
        return X, y, opponent_strengths

    def train_model(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.regressor.fit(X_pca, y)

    def predict_performance(self, features, category):
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        pred_stats = self.regressor.predict(features_pca)[0]
        pred_dict = dict(zip(self.stat_categories, pred_stats))

        if '+' in category:
            stats_needed = category.split('+')
            pred = sum(pred_dict[stat] for stat in stats_needed if stat in pred_dict)
            # Estimate variance for sum using covariance
            residuals = self.regressor.predict(self.pca.transform(self.scaler.transform(X_train))) - y_train
            cov_matrix = np.cov(residuals, rowvar=False)
            indices = [self.stat_categories.index(stat) for stat in stats_needed if stat in self.stat_categories]
            var_sum = sum(cov_matrix[i, j] for i in indices for j in indices)
            sigma = np.sqrt(var_sum) if var_sum > 0 else 1.0
        else:
            pred = pred_dict[category.upper()]
            stat_idx = self.stat_categories.index(category.upper())
            residuals = self.regressor.predict(self.pca.transform(self.scaler.transform(X_train)))[:, stat_idx] - y_train[:, stat_idx]
            sigma = np.std(residuals)

        ci_lower, ci_upper = norm.interval(0.95, loc=pred, scale=sigma)
        return pred, sigma, (ci_lower, ci_upper)

    def predict_over_under(self, player_id, category, opponent_abbr, season_type, betting_line, seasons=['2023-24', '2024-25']):
        all_games = self.prepare_data(player_id, seasons, season_type)
        X, y, opponent_strengths = self.create_features(all_games, opponent_abbr, season_type)
        
        global X_train, y_train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_model(X_train, y_train)

        # Predict for upcoming game
        recent_means = {f'recent_form_{stat}': y[stat].rolling(window=5, min_periods=1).mean().iloc[-1] for stat in self.stat_categories}
        upcoming_features = pd.DataFrame({
            'recency_weight': [1.0],
            'same_opponent': [1],
            'is_playoff': [(season_type == 'Playoffs').astype(int)],
            **{f'weighted_{stat}': [y[stat].mean()] for stat in self.stat_categories},
            **{f'opp_strength_{stat}': [opponent_strengths[stat]] for stat in self.stat_categories},
            **{f'opp_interaction_{stat}': [1 * opponent_strengths[stat]] for stat in self.stat_categories},
            **recent_means
        }, index=[0])[self.feature_cols]

        pred, sigma, ci = self.predict_performance(upcoming_features, category)

        # Probability calculation
        p_over = 1 - norm.cdf(betting_line, pred, sigma)
        confidence = p_over if p_over > 0.5 else 1 - p_over
        bet_on = 'over' if p_over > 0.5 else 'under'

        message = (f"Predicted {category}: {pred:.1f} (95% CI: {ci[0]:.1f}-{ci[1]:.1f})\n"
                   f"P(Over {betting_line}): {p_over*100:.1f}%\n"
                   f"{confidence*100:.1f}% confident bet on {bet_on.upper()}")

        return {'bet_on': bet_on, 'confidence': confidence, 'message': message}

# Example usage
# predictor = AdvancedNBAPlayerPredictor()
# result = predictor.predict_over_under(player_id=2544, category='points+rebounds', opponent_abbr='BOS', season_type='Regular Season', betting_line=35.5)
# print(result['message'])
