"""Flask server for NBA Betting Predictor API."""
from flask import Flask, request, jsonify
from flask_cors import CORS
from predictive_model import AdvancedNBAPlayerPredictor
import bet_calculations as bc
import nba_source
from nba_api.stats.static import players, teams
import logging
import os
import time
from urllib.parse import unquote

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Allowed browser origins. Local dev ports are always permitted; a deployed
# frontend adds its own origin via ALLOWED_ORIGINS (comma-separated), e.g.
#   ALLOWED_ORIGINS=https://my-app.vercel.app,https://my-app-git-main.vercel.app
_DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
_extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = _DEFAULT_ORIGINS + _extra_origins
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})
logger.info("CORS allowed origins: %s", ALLOWED_ORIGINS)

# Log the data mode up front. In production this must read "snapshot" -- if it
# says "live", the deployment will hang on stats.nba.com rather than fail loudly.
logger.info("NBA data source: %s", nba_source.describe())

# Initialize predictor
predictor = AdvancedNBAPlayerPredictor()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint to confirm API is running."""
    logger.info("Accessed health check endpoint")
    return jsonify({"status": "healthy", "message": "NBA Betting Predictor API is running", "timestamp": time.time()})

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """Return empty response for favicon requests."""
    return '', 204

@app.route('/ws', methods=['GET'])
def ws_ignore():
    """Ignore WebSocket requests to reduce log noise."""
    logger.warning("WebSocket request received but not implemented, returning 404")
    return '', 404

@app.route('/api/all-players', methods=['GET'])
def get_all_players():
    """API endpoint to get all active NBA players."""
    try:
        start_time = time.time()
        player_list = bc.get_all_active_players()
        if not player_list:
            logger.warning("No players returned from get_all_active_players")
            return jsonify({"error": "No active players available"}), 503
        logger.info(f"Fetched {len(player_list)} players in {time.time() - start_time}s")
        return jsonify(player_list)
    except Exception as e:
        logger.error(f"Error in /api/all-players: {str(e)}")
        return jsonify({"error": f"Service unavailable: {str(e)}"}), 503

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """API endpoint to get list of NBA teams."""
    try:
        start_time = time.time()
        nba_teams = teams.get_teams()
        logger.info(f"Fetched {len(nba_teams)} teams in {time.time() - start_time}s")
        team_list = [{'id': team['id'], 'abbreviation': team['abbreviation'], 'full_name': team['full_name']} for team in nba_teams]
        return jsonify(team_list)
    except Exception as e:
        logger.error(f"Error fetching teams: {str(e)}")
        return jsonify({"error": f"Service unavailable: {str(e)}"}), 503

@app.route('/api/player-details/<player_name>', methods=['GET'])
def get_player_details(player_name):
    """API endpoint to get detailed player information."""
    try:
        # Some hosts hand WSGI a path that has not been percent-decoded, so a
        # name like "Stephen%20Curry" arrives with the escape intact and never
        # matches. Decoding here is a no-op when the server already did it.
        player_name = unquote(player_name)
        player_info = bc.get_player_id(player_name)
        if not player_info:
            logger.error(f"Player not found: {player_name}")
            return jsonify({"error": f"Player '{player_name}' not found"}), 404

        # Fetch detailed player information
        detailed_info = bc.get_player_detailed_info(player_info['player_id'])
        if detailed_info:
            player_info.update(detailed_info)

        logger.info(f"Player details fetched for {player_name}")
        return jsonify(player_info)

    except Exception as e:
        logger.error(f"Error fetching player details for {player_name}: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/predict', methods=['GET'])
def predict():
    """API endpoint to get player prediction and headshot URL."""
    try:
        player_name = request.args.get('player_name')
        category = request.args.get('category')
        opponent_abbr = request.args.get('opponent_abbr')
        betting_line = request.args.get('betting_line')
        season_type = request.args.get('season_type', 'Regular Season')

        if not all([player_name, category, opponent_abbr, betting_line]):
            logger.error("Missing required parameters")
            return jsonify({"error": "Missing required parameters: player_name, category, opponent_abbr, betting_line"}), 400

        if betting_line is None:
            logger.error("betting_line is None")
            return jsonify({"error": "betting_line is required"}), 400
        try:
            betting_line = float(betting_line)
        except (ValueError, TypeError):
            logger.error(f"Invalid betting_line: {betting_line}")
            return jsonify({"error": "betting_line must be a number"}), 400

        valid_categories = ['Points', 'Rebounds', 'Assists', 'Blocks', 'Steals',
                          'Points+Rebounds+Assists', 'Rebounds+Assists',
                          'Points+Rebounds', 'Points+Assists', 'Blocks+Steals']
        if category not in valid_categories:
            logger.error(f"Invalid category: {category}")
            return jsonify({"error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"}), 400

        if season_type not in ['Regular Season', 'Playoffs']:
            logger.error(f"Invalid season_type: {season_type}")
            return jsonify({"error": "season_type must be 'Regular Season' or 'Playoffs'"}), 400

        player_info = bc.get_player_id(player_name)
        if not player_info:
            logger.error(f"Player not found: {player_name}")
            return jsonify({"error": f"Player '{player_name}' not found"}), 404

        # Fetch detailed player information
        detailed_info = bc.get_player_detailed_info(player_info['player_id'])
        if detailed_info:
            player_info.update(detailed_info)

        team_id = bc.get_team_id(opponent_abbr)
        if not team_id:
            logger.error(f"Team not found: {opponent_abbr}")
            return jsonify({"error": f"Team '{opponent_abbr}' not found"}), 404

        # Fetch additional data for prediction; player averages come from the
        # current season, falling back to the previous one inside bet_calculations.
        logger.info(f"Fetching averages for {player_name} using season {bc.CURRENT_SEASON}")
        h2h_stats, h2h_list = bc.get_head_to_head_stats(player_info['player_id'], opponent_abbr)
        averages = bc.get_player_season_recent_averages(player_info['player_id'], bc.CURRENT_SEASON, season_type)
        logger.debug(f"Player averages for {player_name}: {averages}")

        result = predictor.predict_over_under(
            player_id=player_info['player_id'],
            category=category,
            opponent_abbr=opponent_abbr,
            season_type=season_type,
            betting_line=betting_line
        )

        result['headshot_url'] = player_info['headshot_url']
        result['player_name'] = player_name
        result['h2h_list'] = h2h_list
        result['player_averages'] = averages
        # Ensure predicted_value is always present in the response
        if 'predicted_value' not in result:
            result['predicted_value'] = None
        logger.debug(f"Prediction result for {player_name}: {result}")

        logger.info(f"Prediction successful for {player_name}, category {category}, opponent {opponent_abbr}")
        return jsonify(result)

    except ValueError as e:
        # Raised when there's no game history to model. In snapshot mode this
        # most often means the player isn't in the current data snapshot yet --
        # a real, expected state that deserves a clear message rather than a
        # generic 500.
        logger.warning(f"No data to predict for {player_name}: {e}")
        detail = "No game data available for this player."
        if nba_source.SNAPSHOT_MODE:
            detail += " They may not be in the current data snapshot yet."
        return jsonify({"error": detail}), 503

    except Exception as e:
        logger.error(f"Error processing prediction for {player_name}: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    # PORT is what most hosts inject; default to 5001 to match the frontend's
    # local fallback. Debug stays off unless explicitly asked for.
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
