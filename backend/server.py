from flask import Flask, request, jsonify
from flask_cors import CORS
from predictive_model import AdvancedNBAPlayerPredictor
import bet_calculations as bc
from nba_api.stats.static import players, teams
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "https://your-deployed-frontend-url.com"]}})

# Initialize predictor
predictor = AdvancedNBAPlayerPredictor()

@app.route('/', methods=['GET'])
def home():
    """Root endpoint to confirm API is running."""
    logger.info("Accessed root endpoint")
    return jsonify({
        "message": "Welcome to the NBA Betting Predictor API",
        "endpoints": {
            "/api/predict": "Get player prediction and headshot URL",
            "/api/players": "Get list of active players",
            "/api/teams": "Get list of NBA teams"
        }
    })

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """Return empty response for favicon requests."""
    return '', 204

@app.route('/api/predict', methods=['GET'])
def predict():
    """API endpoint to get player prediction and headshot URL."""
    try:
        # Get query parameters
        player_name = request.args.get('player_name')
        category = request.args.get('category')
        opponent_abbr = request.args.get('opponent_abbr')
        betting_line = request.args.get('betting_line')
        season_type = request.args.get('season_type', 'Regular Season')

        # Validate inputs
        if not all([player_name, category, opponent_abbr, betting_line]):
            logger.error("Missing required parameters")
            return jsonify({"error": "Missing required parameters: player_name, category, opponent_abbr, betting_line"}), 400

        try:
            betting_line = float(betting_line)
        except ValueError:
            logger.error(f"Invalid betting_line: {betting_line}")
            return jsonify({"error": "betting_line must be a number"}), 400

        # Validate category
        valid_categories = ['POINTS', 'REBOUNDS', 'ASSISTS', 'BLOCKS', 'STEALS', 
                        'POINTS+REBOUNDS+ASSISTS', 'REBOUNDS+ASSISTS', 
                        'POINTS+REBOUNDS', 'POINTS+ASSISTS', 'BLOCKS+STEALS']
        if category.upper() not in valid_categories:
            logger.error(f"Invalid category: {category}")
            return jsonify({"error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"}), 400

        # Validate season_type
        if season_type not in ['Regular Season', 'Playoffs']:
            logger.error(f"Invalid season_type: {season_type}")
            return jsonify({"error": "season_type must be 'Regular Season' or 'Playoffs'"}), 400

        # Get player ID and headshot URL
        player_info = bc.get_player_id(player_name)
        if not player_info:
            logger.error(f"Player not found: {player_name}")
            return jsonify({"error": f"Player '{player_name}' not found"}), 404

        # Get team ID to validate opponent
        team_id = bc.get_team_id(opponent_abbr)
        if not team_id:
            logger.error(f"Team not found: {opponent_abbr}")
            return jsonify({"error": f"Team '{opponent_abbr}' not found"}), 404

        # Make prediction
        result = predictor.predict_over_under(
            player_id=player_info['player_id'],
            category=category,
            opponent_abbr=opponent_abbr,
            season_type=season_type,
            betting_line=betting_line
        )

        # Add headshot URL and player name to response
        result['headshot_url'] = player_info['headshot_url']
        result['player_name'] = player_name

        logger.info(f"Prediction successful for {player_name}, category {category}, opponent {opponent_abbr}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing prediction for {player_name}: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/players', methods=['GET'])
def get_players():
    """API endpoint to get list of active players."""
    try:
        active_players = players.get_active_players()
        player_list = [
            {
                'id': player['id'],
                'full_name': player['full_name']
            }
            for player in active_players
        ]
        logger.info(f"Retrieved {len(player_list)} active players")
        return jsonify(player_list)
    except Exception as e:
        logger.error(f"Error fetching players: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """API endpoint to get list of NBA teams."""
    try:
        nba_teams = teams.get_teams()
        team_list = [
            {
                'id': team['id'],
                'abbreviation': team['abbreviation'],
                'full_name': team['full_name']
            }
            for team in nba_teams
        ]
        logger.info(f"Retrieved {len(team_list)} teams")
        return jsonify(team_list)
    except Exception as e:
        logger.error(f"Error fetching teams: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
