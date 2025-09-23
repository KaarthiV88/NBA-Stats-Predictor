import React from 'react';

const PredictionScorecard = ({ prediction, category, bettingLine, teamColor = '#3EB489' }) => {
  if (!prediction) {
    return null;
  }

  const { bet_on, confidence, predicted_value, confidence_interval } = prediction;
  const bettingLineNum = parseFloat(bettingLine);
  
  // Format confidence as percentage with one decimal place
  const confidenceFormatted = confidence ? confidence.toFixed(1) : '0.0';
  
  // Determine if it's a defensive category for styling
  // const isDefensiveCategory = ['Blocks', 'Steals', 'Blocks+Steals'].includes(category);
  
  // Calculate the difference between predicted and betting line
  const difference = predicted_value - bettingLineNum;
  const differenceFormatted = difference > 0 ? `+${difference.toFixed(1)}` : difference.toFixed(1);

  return (
    <div className="prediction-scorecard" style={{ borderColor: teamColor }}>
      <div className="scorecard-header">
        <h3 style={{ color: teamColor }}>Prediction Analysis</h3>
      </div>
      
      <div className="scorecard-content">
        <div className="prediction-main">
          <div className="predicted-value-section">
            <div className="predicted-value-label">Predicted {category}</div>
            <div className="predicted-value" style={{ color: teamColor }}>
              {predicted_value}
            </div>
            <div className="confidence-interval">
              (95% CI: {confidence_interval})
            </div>
          </div>
          
          <div className="betting-line-section">
            <div className="betting-line-label">Betting Line</div>
            <div className="betting-line-value">
              {bettingLineNum}
            </div>
          </div>
          
          <div className="difference-section">
            <div className="difference-label">Difference</div>
            <div className={`difference-value ${difference >= 0 ? 'positive' : 'negative'}`}>
              {differenceFormatted}
            </div>
          </div>
        </div>
        
        <div className="recommendation-section">
          <div className="recommendation-header">
            <span className="recommendation-label">Recommendation:</span>
            <span 
              className={`recommendation-bet ${bet_on}`}
              style={{ 
                backgroundColor: bet_on === 'over' ? '#4CAF50' : '#F44336',
                color: 'white'
              }}
            >
              {bet_on?.toUpperCase()}
            </span>
          </div>
          
          <div className="confidence-section">
            <div className="confidence-bar-container">
              <div className="confidence-bar-label">
                Confidence: {confidenceFormatted}%
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ 
                    width: `${Math.min(confidence || 0, 100)}%`,
                    backgroundColor: teamColor
                  }}
                ></div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="analysis-section">
          <div className="analysis-text">
            {bet_on === 'over' ? (
              <p>
                The model predicts <strong>{predicted_value}</strong> {category.toLowerCase()}, 
                which is <strong>{differenceFormatted}</strong> above the betting line of {bettingLineNum}. 
                This suggests a <strong>{confidenceFormatted}%</strong> confidence in the OVER.
              </p>
            ) : (
              <p>
                The model predicts <strong>{predicted_value}</strong> {category.toLowerCase()}, 
                which is <strong>{differenceFormatted}</strong> below the betting line of {bettingLineNum}. 
                This suggests a <strong>{confidenceFormatted}%</strong> confidence in the UNDER.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionScorecard;
