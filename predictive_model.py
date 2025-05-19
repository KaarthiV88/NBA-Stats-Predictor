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
        self.stat_categories = ['PTS', 'REB', 'AST', 'BLK', 'STL']
        self.feature_cols = None
        self.stat_mapping = {
            'POINTS': 'PTS',
            'REBOUNDS': 'REB',
            'ASSISTS': 'AST',
            'BLOCKS': 'BLK',
            'STEALS': 'STL'
        }

    def prepare_data(self, player_id, seasons=['2023-24', '2024-25'], season_type='Regular Season'):
        """
        Fetches and prepares the player's game logs for the specified seasons.
        """
        game_logs = []
        for season in seasons:
            try:
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star=season_type)
                df = gamelog.get_data_frames()[0]
                df['SEASON'] = season
                game_logs.append(df)
            except Exception as e:
                print(f"Error fetching game logs for season {season}: {e}")
                continue
        if not game_logs:
            print(f"No game logs found for player ID {player_id} in specified seasons.")
            return pd.DataFrame(columns=['GAME_DATE', 'MATCHUP', 'SEASON'] + self.stat_categories)
        all_games = pd.concat(game_logs).reset_index(drop=True)
        all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE']).dt.date
        current_date = datetime.now().date()
        all_games['days_ago'] = (current_date - all_games['GAME_DATE']).apply(lambda x: x.days)
        all_games['OPPONENT'] = all_games['MATCHUP'].apply(lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1])
        for stat in self.stat_categories:
            if stat not in all_games.columns:
                all_games[stat] = 0.0
        return all_games

    def create_features(self, all_games, opponent_abbr, season_type, category_type='offensive'):
        """
        Engineers features for the model, including recency weights, opponent strengths, head-to-head averages, and recent opponent stats.
        """
        if len(all_games) < 5:
            raise ValueError("Insufficient game data (less than 5 games) to create reliable features.")
        
        lambda_ = -np.log(0.5) / 30
        all_games['recency_weight'] = np.exp(-lambda_ * all_games['days_ago'])

        try:
            opp_team_id = bc.get_team_id(opponent_abbr)
            opp_def_stats = bc.get_defensive_team_stats(opp_team_id, all_games['SEASON'].iloc[-1] if not all_games.empty else '2023-24', season_type)
            opp_def_stats = opp_def_stats.iloc[0] if isinstance(opp_def_stats, pd.DataFrame) and not opp_def_stats.empty else opp_def_stats
            league_avgs = bc.get_league_defensive_averages(all_games['SEASON'].iloc[-1] if not all_games.empty else '2023-24', season_type)
        except Exception as e:
            print(f"Error fetching opponent stats: {e}")
            opp_def_stats = pd.Series({f'OPP_{stat}': 100.0 for stat in self.stat_categories})
            league_avgs = pd.Series({f'OPP_{stat}': 100.0 for stat in self.stat_categories})

        # Opponent strength
        opponent_strengths = {}
        for stat in self.stat_categories:
            allowed_stat = f'OPP_{stat}'
            opp_allowed = opp_def_stats.get(allowed_stat, np.nan)
            league_allowed = league_avgs.get(allowed_stat, 0)
            opponent_strengths[stat] = opp_allowed / league_allowed if not pd.isna(opp_allowed) and league_allowed != 0 else 1.0

        # Head-to-head averages
        h2h_stats = bc.get_head_to_head_stats(
            player_id=all_games['Player_ID'].iloc[0] if 'Player_ID' in all_games.columns else bc.get_player_id('LeBron James'),  # Fallback
            opponent_abbr=opponent_abbr,
            seasons=['2023-24', '2024-25'],
            season_type=season_type
        )
        h2h_avgs = h2h_stats[self.stat_categories].mean() if not h2h_stats.empty else pd.Series({stat: 0.0 for stat in self.stat_categories})

        # Recent opponent stats (last 10 games)
        measure_type = 'Defense' if category_type == 'offensive' else 'Base'
        opp_recent_stats = bc.get_team_recent_stats(
            team_id=opp_team_id,
            season='2024-25',
            season_type=season_type,
            measure_type=measure_type,
            num_games=10
        )
        stat_prefix = 'OPP_' if measure_type == 'Defense' else ''
        opp_recent_avgs = {stat: opp_recent_stats.get(f'{stat_prefix}{stat}', 100.0) for stat in self.stat_categories}
        opp_win_pct = opp_recent_stats.get('WIN_PCT', 0.5)

        all_games['same_opponent'] = (all_games['OPPONENT'] == opponent_abbr).astype(int)
        all_games['is_playoff'] = 1 if season_type == 'Playoffs' else 0
        for stat in self.stat_categories:
            all_games[f'weighted_{stat}'] = all_games['recency_weight'] * all_games[stat]
            all_games[f'opp_strength_{stat}'] = opponent_strengths[stat]
            all_games[f'opp_interaction_{stat}'] = all_games['same_opponent'] * opponent_strengths[stat]
            all_games[f'recent_form_{stat}'] = all_games[stat].rolling(window=5, min_periods=1).mean()
            all_games[f'h2h_avg_{stat}'] = h2h_avgs[stat]
            all_games[f'opp_recent_{stat}'] = opp_recent_avgs[stat]
        all_games['opp_recent_win_pct'] = opp_win_pct

        self.feature_cols = ['recency_weight', 'same_opponent', 'is_playoff', 'opp_recent_win_pct'] + \
                            [f'weighted_{stat}' for stat in self.stat_categories] + \
                            [f'opp_strength_{stat}' for stat in self.stat_categories] + \
                            [f'opp_interaction_{stat}' for stat in self.stat_categories] + \
                            [f'recent_form_{stat}' for stat in self.stat_categories] + \
                            [f'h2h_avg_{stat}' for stat in self.stat_categories] + \
                            [f'opp_recent_{stat}' for stat in self.stat_categories]

        X = all_games[self.feature_cols].fillna(0)
        y = all_games[self.stat_categories].fillna(0).values
        if y.shape[1] != len(self.stat_categories):
            raise ValueError(f"Target array has incorrect number of columns: {y.shape[1]} vs {len(self.stat_categories)}")
        return X, y, opponent_strengths

    def train_model(self, X, y):
        """
        Trains the Random Forest Regressor model using PCA-transformed features.
        """
        try:
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.regressor.fit(X_pca, y)
        except Exception as e:
            raise ValueError(f"Error training model: {e}")

    def predict_performance(self, features, category):
        """
        Predicts the player's performance for a given category (single or combined).
        """
        try:
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca.transform(features_scaled)
            pred_stats = self.regressor.predict(features_pca)[0]
            pred_dict = dict(zip(self.stat_categories, pred_stats))

            if '+' not in category:
                mapped_category = self.stat_mapping.get(category.upper(), category.upper())
                if mapped_category not in self.stat_categories:
                    raise ValueError(f"Category '{category}' not supported.")
                pred = pred_dict[mapped_category]
                stat_idx = self.stat_categories.index(mapped_category)
                residuals = self.regressor.predict(self.pca.transform(self.scaler.transform(self.X_train)))[:, stat_idx] - self.y_train[:, stat_idx]
                sigma = np.std(residuals) if np.std(residuals) > 0 else 1.0
            else:
                stats_needed = [self.stat_mapping.get(stat.strip().upper(), stat.strip().upper()) for stat in category.split('+')]
                if not all(stat in self.stat_categories for stat in stats_needed):
                    raise ValueError(f"One or more stats in '{category}' not supported.")
                pred = sum(pred_dict[stat] for stat in stats_needed)
                residuals = self.regressor.predict(self.pca.transform(self.scaler.transform(self.X_train))) - self.y_train
                cov_matrix = np.cov(residuals, rowvar=False)
                indices = [self.stat_categories.index(stat) for stat in stats_needed]
                var_sum = sum(cov_matrix[i, j] for i in indices for j in indices)
                sigma = np.sqrt(var_sum) if var_sum > 0 else 1.0

            ci_lower, ci_upper = norm.interval(0.95, loc=pred, scale=sigma)
            return pred, sigma, (ci_lower, ci_upper)
        except Exception as e:
            raise ValueError(f"Error predicting performance: {e}")

    def predict_over_under(self, player_id, category, opponent_abbr, season_type, betting_line, category_type='offensive', seasons=['2023-24', '2024-25']):
        """
        Predicts the over/under bet outcome for the player's performance.
        """
        try:
            all_games = self.prepare_data(player_id, seasons, season_type)
            if all_games.empty:
                raise ValueError("No game data available to make prediction.")
            all_games['Player_ID'] = player_id  # Add for h2h_stats lookup
            X, y, opponent_strengths = self.create_features(all_games, opponent_abbr, season_type, category_type)
            
            self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            self.train_model(self.X_train, self.y_train)

            # Compute recent means for features
            recent_means = {f'recent_form_{stat}': y[:, self.stat_categories.index(stat)].ravel()[-5:].mean() for stat in self.stat_categories}
            h2h_stats = bc.get_head_to_head_stats(player_id, opponent_abbr, seasons, season_type)
            h2h_avgs = h2h_stats[self.stat_categories].mean() if not h2h_stats.empty else pd.Series({stat: 0.0 for stat in self.stat_categories})
            opp_team_id = bc.get_team_id(opponent_abbr)
            measure_type = 'Defense' if category_type == 'offensive' else 'Base'
            opp_recent_stats = bc.get_team_recent_stats(opp_team_id, '2024-25', season_type, measure_type, num_games=10)
            stat_prefix = 'OPP_' if measure_type == 'Defense' else ''
            opp_recent_avgs = {stat: opp_recent_stats.get(f'{stat_prefix}{stat}', 100.0) for stat in self.stat_categories}
            opp_win_pct = opp_recent_stats.get('WIN_PCT', 0.5)

            upcoming_features = pd.DataFrame({
                'recency_weight': [1.0],
                'same_opponent': [1],
                'is_playoff': [1 if season_type == 'Playoffs' else 0],
                'opp_recent_win_pct': [opp_win_pct],
                **{f'weighted_{stat}': [y[:, self.stat_categories.index(stat)].mean()] for stat in self.stat_categories},
                **{f'opp_strength_{stat}': [opponent_strengths[stat]] for stat in self.stat_categories},
                **{f'opp_interaction_{stat}': [1 * opponent_strengths[stat]] for stat in self.stat_categories},
                **recent_means,
                **{f'h2h_avg_{stat}': [h2h_avgs[stat]] for stat in self.stat_categories},
                **{f'opp_recent_{stat}': [opp_recent_avgs[stat]] for stat in self.stat_categories}
            }, index=[0])[self.feature_cols].fillna(0)

            pred, sigma, ci = self.predict_performance(upcoming_features, category)

            p_over = 1 - norm.cdf(betting_line, pred, sigma)
            confidence = p_over if p_over > 0.5 else 1 - p_over
            bet_on = 'over' if p_over > 0.5 else 'under'

            message = (f"Predicted {category}: {pred:.1f} (95% CI: {ci[0]:.1f}-{ci[1]:.1f})\n"
                       f"P(Over {betting_line}): {p_over*100:.1f}%\n"
                       f"{confidence*100:.1f}% confident bet on {bet_on.upper()}")
            return {'bet_on': bet_on, 'confidence': confidence, 'message': message}
        except Exception as e:
            raise ValueError(f"Error in predict_over_under: {e}")
