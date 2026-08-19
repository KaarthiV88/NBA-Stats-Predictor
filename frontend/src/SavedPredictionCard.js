import React from 'react';
import './SavedPredictionCard.css';
import PredictionScorecard from './PredictionScorecard';

const CATEGORY_TICKER = {
  'Points': 'PTS',
  'Rebounds': 'REB',
  'Assists': 'AST',
  'Blocks': 'BLK',
  'Steals': 'STL',
  'Points+Rebounds+Assists': 'PRA',
  'Rebounds+Assists': 'RA',
  'Points+Rebounds': 'PR',
  'Points+Assists': 'PA',
  'Blocks+Steals': 'BS'
};

const formatDate = (dateString) => {
  const date = new Date(dateString);
  if (isNaN(date)) return '—';
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' · ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const SavedPredictionCard = ({ savedPrediction, onDelete }) => {
  const { player, category, bettingLine, opponent, seasonType, prediction, savedAt } = savedPrediction;

  const accent = player?.team_color || '#3EB489';
  const ticker = CATEGORY_TICKER[category] || category;
  const side = prediction?.bet_on;

  return (
    <article
      className={`position-card market-surface side-${side || 'none'}`}
      style={{ '--accent-team': accent }}
    >
      <header className="position-head">
        <div className="position-portrait">
          <img
            src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${player.id}.png?imwidth=1040&imheight=760`}
            alt={player.full_name}
            className="position-avatar"
            onError={(e) => { e.target.style.visibility = 'hidden'; }}
          />
        </div>

        <div className="position-identity">
          <h3 className="position-name">{player.full_name}</h3>
          <div className="position-contract">
            <span className="position-ticker num">{ticker}</span>
            <span className="position-line num">{bettingLine}</span>
            <span className="position-vs">vs {opponent?.abbreviation || opponent?.full_name || '—'}</span>
          </div>
        </div>

        <button
          className="position-delete"
          onClick={() => onDelete(savedPrediction.id)}
          aria-label={`Remove ${player.full_name} ${category} position`}
          title="Remove position"
        >
          ×
        </button>
      </header>

      <div className="position-tags">
        <span className="chip">{player.team_abbreviation || '—'}</span>
        <span className="chip">{player.position || '—'}</span>
        <span className="chip">{seasonType}</span>
        <span className="position-date">{formatDate(savedAt)}</span>
      </div>

      <PredictionScorecard
        prediction={prediction}
        category={category}
        bettingLine={bettingLine}
        teamColor={accent}
        compact
      />
    </article>
  );
};

export default SavedPredictionCard;
