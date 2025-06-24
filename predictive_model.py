import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
from nba_api.stats.endpoints import playergamelog
import bet_calculations as bc
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AdvancedNBAPlayerPredictor:
    def __init__(self, n_components=0.95):
        self.rf_regressors = {}  # Dictionary to store one RF model per stat
        self.gbr_regressors = {}  # Dictionary to store one GBR model per stat
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.stat_categories = ['PTS', 'REB', 'AST', 'BLK', 'STL']
        self.feature_cols = None
        self.ensemble_weights_rf = {}  # Dictionary to store RF weight per stat
        self.stat_mapping = {
            'POINTS': 'PTS',
            'REBOUNDS': 'REB',
            'ASSISTS': 'AST',
            'BLOCKS': 'BLK',
            'STEALS': 'STL',
            'POINTS+REBOUNDS+ASSISTS': ['PTS', 'REB', 'AST'],
            'REBOUNDS+ASSISTS': ['REB', 'AST'],
            'POINTS+REBOUNDS': ['PTS', 'REB'],
            'POINTS+ASSISTS': ['PTS', 'AST'],
            'BLOCKS+STEALS': ['BLK', 'STL']
        }

    def prepare_data(self, player_id, seasons=['2023-24', '2024-25'], season_type='Regular Season'):
        game_logs = []
        for season in seasons:
            for st in ['Regular Season', 'Playoffs']:
                try:
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season,
                        season_type_all_star=st,
                        timeout=60
                    )
                    df = gamelog.get_data_frames()[0]
                    df['SEASON'] = season
                    df['SEASON_TYPE'] = st
                    game_logs.append(df)
                except Exception as e:
                    logger.warning(f"Error fetching game log for season {season}, type {st}: {e}")
                    continue
        if not game_logs:
            return pd.DataFrame(columns=['GAME_DATE', 'MATCHUP', 'SEASON', 'SEASON_TYPE', 'MIN'] + self.stat_categories)
        all_games = pd.concat(game_logs).reset_index(drop=True)
        all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'], format='mixed').dt.date
        all_games = all_games.sort_values('GAME_DATE', ascending=False)
        current_date = datetime.now().date()
        all_games['days_ago'] = (current_date - all_games['GAME_DATE']).apply(lambda x: x.days)
        all_games['OPPONENT'] = all_games['MATCHUP'].apply(lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1])
        all_games['HOME_AWAY'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        # Filter outliers
        all_games = all_games[(all_games['PTS'] <= 50.0) & (all_games['MIN'] >= 10.0)]
        for stat in self.stat_categories + ['MIN']:
            all_games[stat] = all_games.get(stat, 0.0)
        return all_games

    def create_features(self, all_games, opponent_abbr, season_type, category_type='offensive'):
        if len(all_games) < 5:
            raise ValueError("Insufficient game data (less than 5 games) to create reliable features")
        
        lambda_ = -np.log(0.5) / 30
        all_games['recency_weight'] = np.exp(-lambda_ * all_games['days_ago'])

        try:
            opp_team_id = bc.get_team_id(opponent_abbr)
            opp_def_stats = bc.get_defensive_team_stats(opp_team_id, season='2024-25', season_type=season_type)
            league_avgs = bc.get_league_defensive_averages(season='2024-25', season_type=season_type)
        except Exception as e:
            logger.warning(f"Error fetching opponent stats: {e}")
            opp_def_stats = pd.Series({stat: 110.0 for stat in self.stat_categories})
            league_avgs = pd.Series({stat: 110.0 for stat in self.stat_categories})

        opponent_strengths = {}
        for stat in self.stat_categories:
            opp_allowed = opp_def_stats.get(stat, 110.0)
            league_allowed = league_avgs.get(stat, 110.0)
            opponent_strengths[stat] = opp_allowed / league_allowed if league_allowed != 0 else 1.0

        player_id = all_games['Player_ID'].iloc[0] if 'Player_ID' in all_games.columns else bc.get_player_id('LeBron James')
        h2h_stats, h2h_games = bc.get_head_to_head_stats(player_id, opponent_abbr, seasons=['2023-24', '2024-25'])
        h2h_avgs = h2h_stats[self.stat_categories].mean().fillna(0) if not h2h_stats.empty else pd.Series({stat: 0.0 for stat in self.stat_categories})

        measure_type = 'Defense' if category_type == 'offensive' else 'Base'
        opp_recent_stats = bc.get_team_recent_stats(opp_team_id, '2024-25', season_type, measure_type, num_games=10)
        opp_recent_avgs = {
            stat: opp_recent_stats.get(stat, 112.0 if measure_type == 'Base' else 110.0)
            for stat in self.stat_categories
        }
        opp_pace = opp_recent_stats.get('PACE', 100.0)
        opp_rating = opp_recent_stats.get('OFF_RATING' if measure_type == 'Base' else 'DEF_RATING', 112.0 if measure_type == 'Base' else 110.0)
        opp_win_pct = opp_recent_stats.get('WIN_PCT', 0.5)
        opp_home_away = 0.5

        advanced_stats = bc.get_player_advanced_stats(player_id, '2024-25', season_type)
        usg_pct = advanced_stats.get('USG_PCT', 0.2)
        ts_pct = advanced_stats.get('TS_PCT', 0.5)

        fatigue_metrics = bc.get_player_fatigue_metrics(player_id, '2024-25', season_type, num_games=10)
        avg_min = fatigue_metrics.get('AVG_MIN', 30.0)
        avg_rest_days = fatigue_metrics.get('AVG_REST_DAYS', 2.0)
        # Add injury/rest risk feature
        injury_risk = 1 if avg_min > 38.0 or avg_rest_days < 1.0 else 0

        # Use exponential moving average for trends
        recent_games = all_games.head(10)
        ema_trends = {}
        for stat in self.stat_categories:
            if not recent_games.empty:
                ema = recent_games[stat].ewm(span=5, adjust=False).mean().iloc[-1]
                ema_trends[stat] = ema - recent_games[stat].mean() if len(recent_games) > 1 else 0.0
            else:
                ema_trends[stat] = 0.0

        all_games['same_opponent'] = (all_games['OPPONENT'] == opponent_abbr).astype(int)
        all_games['is_playoff'] = (all_games['SEASON_TYPE'] == 'Playoffs').astype(int)
        for stat in self.stat_categories:
            all_games[f'weighted_{stat}'] = all_games['recency_weight'] * all_games[stat]
            all_games[f'opp_strength_{stat}'] = opponent_strengths[stat]
            all_games[f'opp_interaction_{stat}'] = all_games['same_opponent'] * opponent_strengths[stat]
            all_games[f'recent_form_{stat}'] = all_games[stat].rolling(window=5, min_periods=1).mean().fillna(0)
            all_games[f'h2h_avg_{stat}'] = h2h_avgs[stat]
            all_games[f'opp_recent_{stat}'] = opp_recent_avgs[stat]
            all_games[f'ema_trend_{stat}'] = ema_trends[stat]
        all_games['opp_pace'] = opp_pace
        all_games['opp_rating'] = opp_rating
        all_games['opp_win_pct'] = opp_win_pct
        all_games['opp_home_away'] = opp_home_away
        all_games['usg_pct'] = usg_pct
        all_games['ts_pct'] = ts_pct
        all_games['avg_min'] = avg_min
        all_games['avg_rest_days'] = avg_rest_days
        all_games['injury_risk'] = injury_risk

        self.feature_cols = [
            'recency_weight', 'same_opponent', 'is_playoff', 'opp_pace', 'opp_rating', 'opp_win_pct',
            'opp_home_away', 'usg_pct', 'ts_pct', 'avg_min', 'avg_rest_days', 'injury_risk', 'HOME_AWAY'
        ] + [f'weighted_{stat}' for stat in self.stat_categories] + \
          [f'opp_strength_{stat}' for stat in self.stat_categories] + \
          [f'opp_interaction_{stat}' for stat in self.stat_categories] + \
          [f'recent_form_{stat}' for stat in self.stat_categories] + \
          [f'h2h_avg_{stat}' for stat in self.stat_categories] + \
          [f'opp_recent_{stat}' for stat in self.stat_categories] + \
          [f'ema_trend_{stat}' for stat in self.stat_categories]

        X = all_games[self.feature_cols].fillna(0)
        y = all_games[self.stat_categories].fillna(0).values
        if y.shape[1] != len(self.stat_categories):
            raise ValueError(f"Target array has incorrect number of columns: {y.shape[1]} vs {len(self.stat_categories)}")
        
        logger.debug(f"Features created: {self.feature_cols}")
        return X, y, opponent_strengths, h2h_games, h2h_stats, opp_pace, usg_pct, avg_rest_days, avg_min, ts_pct

    def train_model(self, X, y):
        try:
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)

            for stat_idx, stat in enumerate(self.stat_categories):
                y_stat = y[:, stat_idx]

                # Hyperparameter tuning for RandomForest
                rf_param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5]
                }
                rf = GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    rf_param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                rf.fit(X_pca, y_stat)
                self.rf_regressors[stat] = rf.best_estimator_
                logger.debug(f"Best RF params for {stat}: {rf.best_params_}")

                # Hyperparameter tuning for GradientBoosting
                gbr_param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1]
                }
                gbr = GridSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    gbr_param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                gbr.fit(X_pca, y_stat)
                self.gbr_regressors[stat] = gbr.best_estimator_
                logger.debug(f"Best GBR params for {stat}: {gbr.best_params_}")

                # Optimize ensemble weights
                rf_preds = self.rf_regressors[stat].predict(X_pca)
                gbr_preds = self.gbr_regressors[stat].predict(X_pca)
                best_weight = 0.5
                best_mse = float('inf')
                for w in np.linspace(0, 1, 21):  # Finer grid
                    ensemble_preds = w * rf_preds + (1 - w) * gbr_preds
                    mse = np.mean((ensemble_preds - y_stat) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_weight = w
                self.ensemble_weights_rf[stat] = best_weight
                logger.debug(f"Ensemble weight RF for {stat}: {best_weight}")
        except Exception as e:
            raise ValueError(f"Error training model: {e}")

    def predict_performance(self, features, category, season_type, season_avgs, recent_avgs, h2h_avgs, opp_pace, usg_pct, avg_rest_days):
        try:
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca.transform(features_scaled)

            stats_needed = self.stat_mapping.get(category.upper(), [category.upper()])
            if isinstance(stats_needed, str):
                stats_needed = [stats_needed]
            if not all(stat in self.stat_categories for stat in stats_needed):
                raise ValueError(f"One or more stats in '{category}' not supported")

            model_pred = 0.0
            pred_stats = {}
            for stat in self.stat_categories:
                rf_pred = self.rf_regressors[stat].predict(features_pca)[0]
                gbr_pred = self.gbr_regressors[stat].predict(features_pca)[0]
                ensemble_pred = self.ensemble_weights_rf[stat] * rf_pred + (1 - self.ensemble_weights_rf[stat]) * gbr_pred
                pred_stats[stat] = ensemble_pred
                if stat in stats_needed:
                    model_pred += ensemble_pred

            season_avg = sum(season_avgs.get(stat, 0.0) for stat in stats_needed)
            recent_avg = sum(recent_avgs.get(stat, 0.0) for stat in stats_needed)
            h2h_avg = sum(h2h_avgs.get(stat, 0.0) for stat in stats_needed)
            other_factors = usg_pct * 10 + opp_pace / 100 - avg_rest_days / 2
            # Adjust prior weights based on H2H sample size
            h2h_weight = 0.3 if len(h2h_avgs) > 5 else 0.2
            prior_pred = (0.4 * season_avg + 0.3 * recent_avg + h2h_weight * h2h_avg + 0.1 * other_factors)
            pred = 0.6 * model_pred + 0.4 * prior_pred

            residuals = []
            X_train_scaled = self.scaler.transform(self.X_train)
            X_train_pca = self.pca.transform(X_train_scaled)
            for stat in stats_needed:
                stat_idx = self.stat_categories.index(stat)
                rf_train_preds = self.rf_regressors[stat].predict(X_train_pca)
                gbr_train_preds = self.gbr_regressors[stat].predict(X_train_pca)
                ensemble_train_preds = self.ensemble_weights_rf[stat] * rf_train_preds + (1 - self.ensemble_weights_rf[stat]) * gbr_train_preds
                residuals.append(ensemble_train_preds - self.y_train[:, stat_idx])
            residuals = np.sum(residuals, axis=0)
            sigma = np.std(residuals) if np.std(residuals) > 0 else 1.0

            stat_variances = [
                np.var(self.y_train[:, self.stat_categories.index(stat)]) if len(self.y_train) > 1 else 1.0
                for stat in stats_needed
            ]
            sigma *= (0.5 + 0.5 * min(sum(stat_variances) / (10 * len(stats_needed)), 1.0))
            sigma *= 1.1 if season_type == 'Playoffs' else 0.8  # Reduced playoff multiplier
            sigma = max(sigma, 1.0 if len(stats_needed) == 1 else 1.5)

            max_ci_width = sum({'PTS': 5.0, 'REB': 4.0, 'AST': 3.0, 'BLK': 1.5, 'STL': 1.5}.get(stat, 3.0) for stat in stats_needed)
            sigma = min(sigma, max_ci_width / 1.96)

            ci_lower, ci_upper = norm.interval(0.95, loc=pred, scale=sigma)
            return pred, sigma, (ci_lower, ci_upper)
        except Exception as e:
            raise ValueError(f"Error predicting performance: {e}")

    def predict_over_under(self, player_id, category, opponent_abbr, season_type, betting_line, category_type='offensive', seasons=['2023-24', '2024-25']):
        try:
            all_games = self.prepare_data(player_id, seasons, season_type)
            if all_games.empty:
                raise ValueError("No game data available to make prediction")
            all_games['Player_ID'] = player_id
            
            X, y, opponent_strengths, h2h_games, h2h_stats, opp_pace, usg_pct, avg_rest_days, avg_min, ts_pct = self.create_features(
                all_games, opponent_abbr, season_type, category_type
            )
            
            self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            self.train_model(self.X_train, self.y_train)

            averages = bc.get_player_season_recent_averages(player_id, '2024-25', season_type)
            season_avgs = averages['season_averages']
            recent_avgs = averages['recent_averages']
            h2h_avgs = h2h_stats[self.stat_categories].mean().fillna(0) if not h2h_stats.empty else pd.Series({stat: 0.0 for stat in self.stat_categories})

            opp_team_id = bc.get_team_id(opponent_abbr)
            measure_type = 'Defense' if category_type == 'offensive' else 'Base'
            opp_recent_stats = bc.get_team_recent_stats(opp_team_id, '2024-25', season_type, measure_type, num_games=10)
            opp_avgs = {
                'PTS': opp_recent_stats.get('PTS', 112.0 if measure_type == 'Base' else 110.0),
                'REB': opp_recent_stats.get('REB', 43.0),
                'AST': opp_recent_stats.get('AST', 25.0),
                'BLK': opp_recent_stats.get('BLK', 5.0),
                'STL': opp_recent_stats.get('STL', 7.0),
                'RATING': opp_recent_stats.get('OFF_RATING' if measure_type == 'Base' else 'DEF_RATING', 112.0 if measure_type == 'Base' else 110.0)
            }
            # Validate Defensive Rating
            if opp_avgs['RATING'] < 50.0 or opp_avgs['RATING'] > 150.0:
                logger.warning(f"Implausible opp_avgs['RATING']: {opp_avgs['RATING']}, using default 110.0")
                opp_avgs['RATING'] = 110.0

            recent_means = {
                f'recent_form_{stat}': y[:, self.stat_categories.index(stat)].ravel()[-5:].mean()
                for stat in self.stat_categories
            }
            opp_recent_avgs = {
                stat: opp_recent_stats.get(stat, 112.0 if measure_type == 'Base' else 110.0)
                for stat in self.stat_categories
            }
            opp_win_pct = opp_recent_stats.get('WIN_PCT', 0.5)
            opp_home_away = 0.5

            recent_games = all_games.head(10)
            ema_trends = {}
            for stat in self.stat_categories:
                if not recent_games.empty:
                    ema = recent_games[stat].ewm(span=5, adjust=False).mean().iloc[-1]
                    ema_trends[stat] = ema - recent_games[stat].mean() if len(recent_games) > 1 else 0.0
                else:
                    ema_trends[stat] = 0.0

            upcoming_features = pd.DataFrame({
                'recency_weight': [1.0],
                'same_opponent': [1],
                'is_playoff': [1 if season_type == 'Playoffs' else 0],
                'opp_pace': [opp_pace],
                'opp_rating': [opp_avgs['RATING']],
                'opp_win_pct': [opp_win_pct],
                'opp_home_away': [opp_home_away],
                'usg_pct': [usg_pct],
                'ts_pct': [ts_pct],
                'avg_min': [avg_min],
                'avg_rest_days': [avg_rest_days],
                'injury_risk': [1 if avg_min > 38.0 or avg_rest_days < 1.0 else 0],
                'HOME_AWAY': [1],
                **{f'weighted_{stat}': [y[:, self.stat_categories.index(stat)].mean()] for stat in self.stat_categories},
                **{f'opp_strength_{stat}': [opponent_strengths[stat]] for stat in self.stat_categories},
                **{f'opp_interaction_{stat}': [1 * opponent_strengths[stat]] for stat in self.stat_categories},
                **{f'recent_form_{stat}': [recent_means[f'recent_form_{stat}']] for stat in self.stat_categories},
                **{f'h2h_avg_{stat}': [h2h_avgs[stat]] for stat in self.stat_categories},
                **{f'opp_recent_{stat}': [opp_recent_avgs[stat]] for stat in self.stat_categories},
                **{f'ema_trend_{stat}': [ema_trends[stat]] for stat in self.stat_categories}
            }, index=[0])[self.feature_cols].fillna(0)

            pred, sigma, ci = self.predict_performance(
                upcoming_features, category, season_type, season_avgs, recent_avgs, h2h_avgs, opp_pace, usg_pct, avg_rest_days
            )

            p_over = 1 - norm.cdf(betting_line, pred, sigma)
            confidence = p_over if p_over > 0.5 else 1 - p_over
            bet_on = 'over' if p_over > 0.5 else 'under'

            stats_needed = self.stat_mapping.get(category.upper(), [category.upper()])
            if isinstance(stats_needed, str):
                stats_needed = [stats_needed]
            season_avg = sum(season_avgs.get(stat, 0.0) for stat in stats_needed)
            recent_avg = sum(recent_avgs.get(stat, 0.0) for stat in stats_needed)
            h2h_avg = sum(h2h_avgs.get(stat, 0.0) for stat in stats_needed)

            stat_type = 'Defensive' if category_type == 'offensive' else 'Offensive'
            message = f"Head-to-Head Matchups vs. {opponent_abbr}:\n"
            if h2h_games:
                for game in h2h_games:
                    message += (f"  Date: {game['Game_Date']}, Matchup: {game['Matchup']}, "
                               f"PTS: {game['PTS']:.1f}, REB: {game['REB']:.1f}, AST: {game['AST']:.1f}, "
                               f"BLK: {game['BLK']:.1f}, STL: {game['STL']:.1f}\n")
            else:
                message += "  No matchups found.\n"
            message += (f"\nPlayer Averages for {category}:\n"
                       f"  Season: {season_avg:.1f}\n"
                       f"  Last 10 Games: {recent_avg:.1f}\n"
                       f"  vs. {opponent_abbr} (Regular + Playoffs): {h2h_avg:.1f}\n"
                       f"\nOpponent {stat_type} Averages (Last 10 Games):\n"
                       f"  Points: {opp_avgs['PTS']:.1f}\n"
                       f"  Rebounds: {opp_avgs['REB']:.1f}\n"
                       f"  Assists: {opp_avgs['AST']:.1f}\n"
                       f"  Blocks: {opp_avgs['BLK']:.1f}\n"
                       f"  Steals: {opp_avgs['STL']:.1f}\n"
                       f"  {stat_type} Rating: {opp_avgs['RATING']:.1f}\n"
                       f"\nPredicted {category}: {pred:.1f} (95% CI: {ci[0]:.1f}-{ci[1]:.1f})\n"
                       f"P(Over {betting_line}): {p_over*100:.1f}%\n"
                       f"{confidence*100:.1f}% confident bet on {bet_on.upper()}")
            return {'bet_on': bet_on, 'confidence': confidence, 'message': message}
        except Exception as e:
            raise ValueError(f"Error in predict_over_under: {e}")
