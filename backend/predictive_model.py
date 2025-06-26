import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.sparse import csr_matrix
from nba_api.stats.endpoints import leaguedashteamstats
import bet_calculations as bc
import logging
from joblib import Parallel, delayed
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNBAPlayerPredictor:
    def __init__(self, n_components=0.95):
        self.gb_regressors = {}
        self.rf_regressors = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.stat_categories = ['PTS', 'REB', 'AST', 'BLK', 'STL']
        self.feature_cols = None
        self.ensemble_weights_rf = {}
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
        self.opponent_stats_cache = {}
        self.model_cache = {}

    def compute_data_hash(self, all_games):
        """Compute SHA-256 hash of game log data."""
        data_str = all_games.to_json()
        return hashlib.sha256(data_str.encode()).hexdigest()

    def prepare_data(self, player_id, seasons=['2023-24', '2024-25'], season_type='Regular Season'):
        game_logs = []
        for season in seasons:
            for st in ['Regular Season', 'Playoffs']:
                try:
                    gamelog = bc.playergamelog.PlayerGameLog(
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
            logger.warning(f"No game logs found for player_id {player_id}")
            return pd.DataFrame(columns=['GAME_DATE', 'MATCHUP', 'SEASON', 'SEASON_TYPE', 'MIN'] + self.stat_categories)
        all_games = pd.concat(game_logs).reset_index(drop=True)
        all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'], format='mixed', errors='coerce').dt.date
        all_games = all_games.sort_values('GAME_DATE', ascending=True)
        current_date = datetime.now().date()
        all_games['days_ago'] = (current_date - all_games['GAME_DATE']).apply(lambda x: x.days)
        all_games['OPPONENT'] = all_games['MATCHUP'].apply(
            lambda x: x.split(' vs. ')[1] if ' vs. ' in x else x.split(' @ ')[1]
        )
        all_games['HOME_AWAY'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        all_games = all_games[(all_games['PTS'] <= 50.0) & (all_games['MIN'] >= 10.0)]
        for stat in self.stat_categories + ['MIN']:
            all_games[stat] = all_games.get(stat, 0.0)
        return all_games

    def fetch_all_team_stats(self, seasons, season_type):
        """Batch fetch team stats for all teams."""
        for season in seasons:
            if (season, season_type) not in self.opponent_stats_cache:
                try:
                    team_stats = leaguedashteamstats.LeagueDashTeamStats(
                        season=season,
                        season_type_all_star=season_type,
                        measure_type_detailed_defense='Defense',
                        last_n_games=82,
                        timeout=60
                    ).get_data_frames()[0]
                    for _, row in team_stats.iterrows():
                        team_id = row['TEAM_ID']
                        games_played = row.get('GP', 82)
                        stats = {
                            f'opp_{stat}': row.get(api_stat, default_val) / games_played
                            for stat, api_stat, default_val in [
                                ('PTS', 'OPP_PTS', 110.0), ('REB', 'DREB', 43.0), ('AST', 'OPP_AST', 25.0),
                                ('BLK', 'BLK', 5.0), ('STL', 'STL', 7.0)
                            ]
                        }
                        stats.update({
                            'PACE': row.get('PACE', 100.0),
                            'DEF_RATING': row.get('DEF_RATING', 110.0)
                        })
                        self.opponent_stats_cache[(team_id, season)] = pd.Series(stats)
                    logger.debug(f"Fetched team stats for season {season}")
                except Exception as e:
                    logger.warning(f"Error fetching team stats for season {season}: {e}")

    def compute_h2h_features(self, games, stat):
        """Compute H2H averages using prefix sums."""
        if games.empty:
            return np.zeros(len(games))
        cumsum = games[stat].cumsum().shift(1).fillna(0)
        counts = np.arange(len(games)).astype(float)
        counts[0] = 1
        return cumsum / counts

    def create_features(self, all_games, opponent_abbr, season_type, category_type='offensive'):
        if len(all_games) < 5:
            raise ValueError("Insufficient game data (less than 5 games) to create reliable features")

        lambda_ = -np.log(0.5) / 30
        all_games['recency_weight'] = np.exp(-lambda_ * all_games['days_ago'])

        seasons = all_games['SEASON'].unique()
        self.fetch_all_team_stats(seasons, 'Regular Season')
        league_avgs = {season: bc.get_league_defensive_averages(season, 'Regular Season') for season in seasons}
        for season in league_avgs:
            league_avgs[season] = pd.Series({
                f'opp_{k}': v for k, v in league_avgs[season].items() if k in self.stat_categories
            }).combine_first(pd.Series({
                'PACE': 100.0,
                'DEF_RATING': 110.0
            }))

        opponent_stats = {}
        for opp in all_games['OPPONENT'].unique():
            try:
                team_id = bc.get_team_id(opp)
                for season in seasons:
                    cached_stats = self.opponent_stats_cache.get((team_id, season))
                    if cached_stats is not None:
                        opponent_stats[(opp, season)] = cached_stats
                    else:
                        team_stats = bc.get_team_recent_stats(
                            team_id, season, 'Regular Season', 'Defense', num_games=82
                        )
                        opponent_stats[(opp, season)] = pd.Series({
                            f'opp_{stat}': team_stats.get(stat, league_avgs[season][f'opp_{stat}'])
                            for stat in self.stat_categories
                        }).combine_first(pd.Series({
                            'PACE': team_stats.get('PACE', 100.0),
                            'DEF_RATING': team_stats.get('DEF_RATING', 110.0)
                        }))
                        self.opponent_stats_cache[(team_id, season)] = opponent_stats[(opp, season)]
            except Exception as e:
                logger.warning(f"Error fetching stats for {opp}, season {season}: {e}")
                for season in seasons:
                    opponent_stats[(opp, season)] = league_avgs[season]

        opp_data = pd.DataFrame([
            {
                'OPPONENT': opp,
                'SEASON': season,
                **opponent_stats[(opp, season)].to_dict()
            }
            for opp, season in [(row['OPPONENT'], row['SEASON']) for _, row in all_games.iterrows()]
        ], index=all_games.index)
        logger.debug(f"opp_data columns: {opp_data.columns}")
        all_games = all_games.join(opp_data.drop(columns=['OPPONENT', 'SEASON']))
        logger.debug(f"all_games columns after join: {all_games.columns}")

        for stat in self.stat_categories:
            all_games[f'opp_strength_{stat}'] = all_games[f'opp_{stat}'] / league_avgs['2024-25'][f'opp_{stat}']
        all_games['opp_pace'] = all_games.get('PACE', 100.0)
        all_games['opp_rating'] = all_games.get('DEF_RATING', 110.0)

        def compute_h2h_opp(opp):
            mask = all_games['OPPONENT'] == opp
            games = all_games[mask].sort_values('GAME_DATE')
            return {stat: (games.index, self.compute_h2h_features(games, stat)) for stat in self.stat_categories}

        h2h_results = Parallel(n_jobs=-1)(delayed(compute_h2h_opp)(opp) for opp in all_games['OPPONENT'].unique())
        h2h_avgs = {stat: pd.Series(0.0, index=all_games.index) for stat in self.stat_categories}
        for result in h2h_results:
            for stat, (idx, values) in result.items():
                h2h_avgs[stat].loc[idx] = values
        for stat in self.stat_categories:
            all_games[f'h2h_avg_{stat}'] = h2h_avgs[stat]
        all_games['has_h2h'] = (all_games[[f'h2h_avg_{stat}' for stat in self.stat_categories]].sum(axis=1) > 0).astype(int)

        player_id = all_games['Player_ID'].iloc[0]
        advanced_stats = bc.get_player_advanced_stats(player_id, '2024-25', season_type)
        usg_pct = advanced_stats.get('USG_PCT', 0.2)
        ts_pct = advanced_stats.get('TS_PCT', 0.5)
        fatigue_metrics = bc.get_player_fatigue_metrics(player_id, '2024-25', season_type, num_games=10)
        avg_min = fatigue_metrics.get('AVG_MIN', 30.0)
        avg_rest_days = fatigue_metrics.get('AVG_REST_DAYS', 2.0)
        injury_risk = 1 if avg_min > 38.0 or avg_rest_days < 1.0 else 0

        recent_games = all_games.tail(10)
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
            all_games[f'opp_interaction_{stat}'] = all_games['same_opponent'] * all_games[f'opp_strength_{stat}']
            all_games[f'recent_form_{stat}'] = all_games[stat].rolling(window=5, min_periods=1).mean().fillna(0)
            all_games[f'opp_recent_{stat}'] = all_games[f'opp_strength_{stat}'] * 110.0
            all_games[f'ema_trend_{stat}'] = ema_trends[stat]
        all_games['usg_pct'] = usg_pct
        all_games['ts_pct'] = ts_pct
        all_games['avg_min'] = avg_min
        all_games['avg_rest_days'] = avg_rest_days
        all_games['injury_risk'] = injury_risk

        self.feature_cols = [
            'recency_weight', 'same_opponent', 'is_playoff', 'opp_pace', 'opp_rating', 'usg_pct', 'ts_pct',
            'avg_min', 'avg_rest_days', 'injury_risk', 'HOME_AWAY', 'has_h2h'
        ] + [f'weighted_{stat}' for stat in self.stat_categories] + \
          [f'opp_strength_{stat}' for stat in self.stat_categories] + \
          [f'opp_interaction_{stat}' for stat in self.stat_categories] + \
          [f'recent_form_{stat}' for stat in self.stat_categories] + \
          [f'h2h_avg_{stat}' for stat in self.stat_categories] + \
          [f'opp_recent_{stat}' for stat in self.stat_categories] + \
          [f'ema_trend_{stat}' for stat in self.stat_categories]

        X = all_games[self.feature_cols].fillna(0)
        variances = X.var()
        self.feature_cols = variances[variances > 0.01].index.tolist()
        X = X[self.feature_cols]
        X_sparse = csr_matrix(X.values)
        y = all_games[self.stat_categories].fillna(0).values
        return {
            'X_sparse': X_sparse,
            'y': y,
            'opponent_stats': opponent_stats,
            'usg_pct': usg_pct,
            'avg_rest_days': avg_rest_days,
            'avg_min': avg_min,
            'ts_pct': ts_pct
        }

    def train_stat_model(self, stat_idx, stat, X_scaled, X_pca, y, data_hash):
        """Train model for a single stat."""
        logger.debug(f"Training model for stat {stat}, stat_idx {stat_idx}, X_scaled shape {X_scaled.shape}, X_pca shape {X_pca.shape}, y shape {y.shape}, data_hash {data_hash}")
        y_stat = y[:, stat_idx]
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
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
        gb_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        gb = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        gb.fit(X_pca, y_stat)
        rf_preds = rf.best_estimator_.predict(X_pca)
        gb_preds = gb.best_estimator_.predict(X_pca)
        best_weight = 0.5
        best_mse = float('inf')
        for w in np.linspace(0, 1, 11):
            ensemble_preds = w * rf_preds + (1 - w) * gb_preds
            mse = np.mean((ensemble_preds - y_stat) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_weight = w
        cache_key = (stat, data_hash)
        self.model_cache[cache_key] = (rf.best_estimator_, gb.best_estimator_, best_weight)
        return stat, rf.best_estimator_, gb.best_estimator_, best_weight

    def train_model(self, X, y, data_hash):
        X_scaled = self.scaler.fit_transform(X.toarray())
        X_pca = self.pca.fit_transform(X_scaled)
        logger.debug(f"Calling Parallel with {len(self.stat_categories)} tasks")
        results = Parallel(n_jobs=-1)(
            delayed(self.train_stat_model)(stat_idx, stat, X_scaled, X_pca, y, data_hash)
            for stat_idx, stat in enumerate(self.stat_categories)
        )
        for stat, rf_model, gb_model, weight in results:
            self.rf_regressors[stat] = rf_model
            self.gb_regressors[stat] = gb_model
            self.ensemble_weights_rf[stat] = weight

    def predict_performance(self, features, category, season_type, season_avgs, recent_avgs, h2h_avgs, opp_pace, usg_pct, avg_rest_days):
        features_scaled = self.scaler.transform(features.toarray())
        features_pca = self.pca.transform(features_scaled)
        stats_needed = self.stat_mapping.get(category.upper(), [category.upper()])
        if isinstance(stats_needed, str):
            stats_needed = [stats_needed]
        model_pred = 0.0
        pred_stats = {}
        for stat in self.stat_categories:
            rf_pred = self.rf_regressors[stat].predict(features_pca)[0]
            gb_pred = self.gb_regressors[stat].predict(features_pca)[0]
            ensemble_pred = self.ensemble_weights_rf[stat] * rf_pred + (1 - self.ensemble_weights_rf[stat]) * gb_pred
            pred_stats[stat] = ensemble_pred
            if stat in stats_needed:
                model_pred += ensemble_pred

        season_avg = sum(season_avgs.get(stat, 0.0) for stat in stats_needed)
        recent_avg = sum(recent_avgs.get(stat, 0.0) for stat in stats_needed)
        h2h_avg = sum(h2h_avgs.get(stat, 0.0) for stat in stats_needed)
        other_factors = usg_pct * 10 + opp_pace / 100 - avg_rest_days / 2
        num_h2h_games = len(h2h_avgs) if isinstance(h2h_avgs, pd.Series) else 0
        if num_h2h_games >= 3:
            weights = {'recent_avg': 0.35, 'h2h_avg': 0.35, 'opp_adjust': 0.15, 'season_avg': 0.15}
        else:
            weights = {'recent_avg': 0.45, 'season_avg': 0.35, 'opp_adjust': 0.15, 'h2h_avg': 0.05}
        prior_pred = (weights['season_avg'] * season_avg + weights['recent_avg'] * recent_avg +
                      weights['h2h_avg'] * h2h_avg + weights['opp_adjust'] * other_factors)
        pred = 0.6 * model_pred + 0.4 * prior_pred

        residuals = []
        X_train_scaled = self.scaler.transform(self.X_train.toarray())
        X_train_pca = self.pca.transform(X_train_scaled)
        for stat in stats_needed:
            stat_idx = self.stat_categories.index(stat)
            rf_train_preds = self.rf_regressors[stat].predict(X_train_pca)
            gb_train_preds = self.gb_regressors[stat].predict(X_train_pca)
            ensemble_train_preds = self.ensemble_weights_rf[stat] * rf_train_preds + (1 - self.ensemble_weights_rf[stat]) * gb_train_preds
            residuals.append(ensemble_train_preds - self.y_train[:, stat_idx])
        residuals = np.sum(residuals, axis=0)
        sigma = np.std(residuals) if np.std(residuals) > 0 else 1.0
        sigma = max(sigma, 1.0 if len(stats_needed) == 1 else 1.5)
        ci_lower, ci_upper = norm.interval(0.95, loc=pred, scale=sigma)
        return pred, sigma, (ci_lower, ci_upper)

    def predict_over_under(self, player_id, category, opponent_abbr, season_type, betting_line, category_type='offensive', seasons=['2023-24', '2024-25']):
        all_games = self.prepare_data(player_id, seasons, season_type)
        if all_games.empty:
            raise ValueError("No game data available to make prediction")
        all_games['Player_ID'] = player_id
        data_hash = self.compute_data_hash(all_games)

        features = self.create_features(all_games, opponent_abbr, season_type, category_type)
        X, y, opponent_strengths, usg_pct, avg_rest_days, avg_min, ts_pct = (
            features['X_sparse'], features['y'], features['opponent_stats'], features['usg_pct'],
            features['avg_rest_days'], features['avg_min'], features['ts_pct']
        )
        self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_model(self.X_train, self.y_train, data_hash)

        averages = bc.get_player_season_recent_averages(player_id, '2024-25', season_type)
        season_avgs = averages['season_averages']
        recent_avgs = averages['recent_averages']
        h2h_stats, h2h_games = bc.get_head_to_head_stats(player_id, opponent_abbr, seasons)
        h2h_avgs = h2h_stats[self.stat_categories].mean().fillna(0) if not h2h_stats.empty else pd.Series({stat: 0.0 for stat in self.stat_categories})

        opp_team_id = bc.get_team_id(opponent_abbr)
        measure_type = 'Defense' if category_type == 'offensive' else 'Base'
        opp_recent_stats = bc.get_team_recent_stats(opp_team_id, '2024-25', season_type, measure_type, num_games=10)
        opp_avgs = {
            'PTS': opp_recent_stats.get('PTS', 110.0),
            'REB': opp_recent_stats.get('REB', 43.0),
            'AST': opp_recent_stats.get('AST', 25.0),
            'BLK': opp_recent_stats.get('BLK', 5.0),
            'STL': opp_recent_stats.get('STL', 7.0),
            'RATING': opp_recent_stats.get('DEF_RATING' if measure_type == 'Defense' else 'OFF_RATING', 110.0)
        }

        recent_means = {f'recent_form_{stat}': y[:, self.stat_categories.index(stat)].ravel()[-5:].mean() for stat in self.stat_categories}
        ema_trends = {}
        recent_games = all_games.tail(10)
        for stat in self.stat_categories:
            if not recent_games.empty:
                ema = recent_games[stat].ewm(span=5, adjust=False).mean().iloc[-1]
                ema_trends[stat] = ema - recent_games[stat].mean() if len(recent_games) > 1 else 0.0
            else:
                ema_trends[stat] = 0.0

        logger.debug(f"opponent_strengths keys: {list(opponent_strengths.keys())}")
        upcoming_features = pd.DataFrame({
            'recency_weight': [1.0],
            'same_opponent': [1],
            'is_playoff': [1 if season_type == 'Playoffs' else 0],
            'opp_pace': [opponent_strengths.get('PACE', 100.0)],
            'opp_rating': [opp_avgs['RATING']],
            'usg_pct': [usg_pct],
            'ts_pct': [ts_pct],
            'avg_min': [avg_min],
            'avg_rest_days': [avg_rest_days],
            'injury_risk': [1 if avg_min > 38.0 or avg_rest_days < 1.0 else 0],
            'HOME_AWAY': [1],
            'has_h2h': [1 if len(h2h_stats) > 0 else 0],
            **{f'weighted_{stat}': [y[:, self.stat_categories.index(stat)].mean()] for stat in self.stat_categories},
            **{f'opp_strength_{stat}': [opponent_strengths.get(f'opp_{stat}', 1.0)] for stat in self.stat_categories},
            **{f'opp_interaction_{stat}': [1 * opponent_strengths.get(f'opp_{stat}', 1.0)] for stat in self.stat_categories},
            **{f'recent_form_{stat}': [recent_means[f'recent_form_{stat}']] for stat in self.stat_categories},
            **{f'h2h_avg_{stat}': [h2h_avgs[stat]] for stat in self.stat_categories},
            **{f'opp_recent_{stat}': [opp_avgs[stat]] for stat in self.stat_categories},
            **{f'ema_trend_{stat}': [ema_trends[stat]] for stat in self.stat_categories}
        }, index=[0])[self.feature_cols].fillna(0)

        upcoming_features_sparse = csr_matrix(upcoming_features.values)
        pred, sigma, ci = self.predict_performance(
            upcoming_features_sparse, category, season_type, season_avgs, recent_avgs, h2h_avgs, opponent_strengths.get('PACE', 100.0), usg_pct, avg_rest_days
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
                   f"  vs. {opponent_abbr}: {h2h_avg:.1f}\n"
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
