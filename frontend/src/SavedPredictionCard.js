import React from 'react';
import './SavedPredictionCard.css';
import PredictionScorecard from './PredictionScorecard';

const SavedPredictionCard = ({ savedPrediction, onDelete, index, total }) => {
  const { 
    player, 
    category, 
    bettingLine, 
    opponent, 
    seasonType, 
    prediction,
    savedAt 
  } = savedPrediction;

  // Extract prediction values
  const predictedValue = (typeof prediction.predicted_value === 'number' && !isNaN(prediction.predicted_value))
    ? prediction.predicted_value.toFixed(1)
    : (prediction.message.match(/Predicted [A-Za-z+]+: (\d+\.\d+)/)?.[1] || 'N/A');
  
  const confidenceInterval = prediction.confidence_interval || (prediction.message.match(/95% CI: (\d+\.\d+-\d+\.\d+)/)?.[1] || 'N/A');
  const confidencePercentage = (typeof prediction.confidence === 'number' && !isNaN(prediction.confidence))
    ? prediction.confidence.toFixed(1)
    : 'N/A';
  const betRecommendation = prediction.bet_on?.toUpperCase() || 'N/A';

  // Parse confidence interval for visual display
  const [lowerBound, upperBound] = confidenceInterval.split('-').map(Number);
  const isConfidenceValid = !isNaN(lowerBound) && !isNaN(upperBound);
  const predictedValueNum = parseFloat(predictedValue);

  // Calculate tick position, clamped to [0, 100]
  let tickPosition = null;
  if (isConfidenceValid && !isNaN(predictedValueNum)) {
    if (upperBound === lowerBound) {
      tickPosition = 50;
    } else {
      const raw = ((predictedValueNum - lowerBound) / (upperBound - lowerBound)) * 100;
      tickPosition = Math.max(0, Math.min(100, raw));
    }
  }

  // Create category abbreviation and format
  const getCategoryDisplay = (cat) => {
    const categoryMap = {
      'Points': 'points',
      'Rebounds': 'rebounds',
      'Assists': 'assists',
      'Blocks': 'blocks',
      'Steals': 'steals',
      'Points+Rebounds+Assists': 'PRA',
      'Rebounds+Assists': 'RA',
      'Points+Rebounds': 'PR',
      'Points+Assists': 'PA',
      'Blocks+Steals': 'BS'
    };
    return categoryMap[cat] || cat.toLowerCase();
  };

  const categoryDisplay = getCategoryDisplay(category);
  const primaryColor = player.team_color || '#3EB489';

  // Format saved date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Removed transform to rely on CSS grid
  const cardStyle = {
    display: 'inline-block',
    margin: '0 1%'
  };

  return (
    <div className="saved-prediction-card" style={{ borderColor: primaryColor, ...cardStyle }}>
      <div className="saved-card-header">
        <div className="saved-player-info">
          <div className="saved-player-image-container">
            <img
              src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${player.id}.png?imwidth=1040&imheight=760`}
              alt={player.full_name}
              className="saved-player-image"
              style={{ borderColor: primaryColor }}
              onError={(e) => { e.target.src = 'https://via.placeholder.com/128'; }}
            />
          </div>
          <div className="saved-player-details">
            <h3 className="saved-player-name" style={{ color: primaryColor }}>
              {player.full_name}
            </h3>
            <div className="saved-player-info-grid">
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">Height:</span>
                <span className="saved-player-info-value">{player.height || 'N/A'}</span>
              </div>
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">Weight:</span>
                <span className="saved-player-info-value">{player.weight ? `${player.weight} lbs` : 'N/A'}</span>
              </div>
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">Jersey:</span>
                <span className="saved-player-info-value">#{player.jersey || 'N/A'}</span>
              </div>
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">Position:</span>
                <span className="saved-player-info-value">{player.position || 'N/A'}</span>
              </div>
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">Team:</span>
                <span className="saved-player-info-value">{player.team_city && player.team_name ? `${player.team_city} ${player.team_name}` : 'N/A'}</span>
              </div>
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">School:</span>
                <span className="saved-player-info-value">{player.school || 'N/A'}</span>
              </div>
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">Country:</span>
                <span className="saved-player-info-value">{player.country || 'N/A'}</span>
              </div>
              <div className="saved-player-info-item">
                <span className="saved-player-info-label">Experience:</span>
                <span className="saved-player-info-value">{player.season_exp ? `${player.season_exp} years` : 'N/A'}</span>
              </div>
            </div>
            <div className="saved-bet-details">
              <span className="saved-bet-category" style={{ backgroundColor: 'rgba(255, 255, 255, 0.15)', borderRadius: '6px', padding: '6px 12px' }}>
                {category}
              </span>
              <span className="saved-bet-line" style={{ backgroundColor: 'rgba(255, 255, 255, 0.15)', borderRadius: '6px', padding: '6px 12px' }}>
                Line: {bettingLine}
              </span>
              <span className="saved-opponent" style={{ backgroundColor: 'rgba(255, 255, 255, 0.15)', borderRadius: '6px', padding: '6px 12px' }}>
                vs {opponent.full_name}
              </span>
              <span className="saved-season-type" style={{ backgroundColor: 'rgba(255, 255, 255, 0.15)', borderRadius: '6px', padding: '6px 12px' }}>
                {seasonType}
              </span>
            </div>
            <div className="saved-date">
              Saved: {formatDate(savedAt)}
            </div>
          </div>
        </div>
        <button 
          className="delete-button"
          onClick={() => onDelete(savedPrediction.id)}
          style={{ backgroundColor: '#dc2626' }}
        >
          ×
        </button>
      </div>

      <div className="saved-prediction-result">
        <PredictionScorecard
          prediction={prediction}
          category={category}
          bettingLine={bettingLine}
          teamColor={primaryColor}
        />
      </div>
    </div>
  );
};

export default SavedPredictionCard;