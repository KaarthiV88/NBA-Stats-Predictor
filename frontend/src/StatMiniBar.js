import React from 'react';
import './StatMiniBar.css';

const STAT_LABELS = {
  PTS: 'Points',
  REB: 'Rebounds',
  AST: 'Assists',
  BLK: 'Blocks',
  STL: 'Steals'
};

// How favourable a matchup reads, worst -> best. The fill width doubles as a
// rating, so a red bar is also a short bar.
const RATING_SCALE = [
  { key: 'poor', label: 'Poor', fill: 20 },
  { key: 'weak', label: 'Weak', fill: 40 },
  { key: 'neutral', label: 'Neutral', fill: 60 },
  { key: 'good', label: 'Good', fill: 80 },
  { key: 'great', label: 'Great', fill: 100 }
];

const StatMiniBar = ({ stat, value, category, title }) => {
  // NBA-contextual thresholds for each stat (opponent's per-game allowances).
  const thresholds = {
    PTS: [109, 112, 115, 118],
    REB: [40, 41.9, 43.5, 45.9],
    AST: [20, 21.9, 25.5, 25.9],
    BLK: [4.0, 4.4, 5.1, 5.4],
    STL: [7.0, 7.4, 7.9, 8.4]
  };

  // Determine stat context (offensive/defensive)
  const offensiveStats = ['Points', 'Assists', 'Rebounds', 'Points+Rebounds+Assists', 'Rebounds+Assists', 'Points+Rebounds', 'Points+Assists'];
  const defensiveStats = ['Blocks', 'Steals', 'Blocks+Steals'];
  const isOffensive = offensiveStats.some(cat => category && category.includes(cat));
  const isDefensive = defensiveStats.some(cat => category && category.includes(cat));

  // Get rating index based on value and context
  const getRatingIndex = (statKey, val) => {
    const t = thresholds[statKey];
    if (!t) return 2; // default to neutral

    // --- OFFENSIVE CATEGORY LOGIC ---
    if (isOffensive) {
      if (statKey === 'PTS' || statKey === 'AST') {
        // Higher is better (greener)
        if (val < t[0]) return 0;
        if (val < t[1]) return 1;
        if (val < t[2]) return 2;
        if (val < t[3]) return 3;
        return 4;
      } else if (statKey === 'REB' || statKey === 'BLK' || statKey === 'STL') {
        // Higher is worse (redder)
        if (val >= t[3]) return 0;
        if (val >= t[2]) return 1;
        if (val >= t[1]) return 2;
        if (val >= t[0]) return 3;
        return 4;
      }
    }
    // --- DEFENSIVE CATEGORY LOGIC ---
    if (isDefensive) {
      if (statKey === 'PTS' || statKey === 'AST') {
        // Higher is worse (redder)
        if (val >= t[3]) return 0;
        if (val >= t[2]) return 1;
        if (val >= t[1]) return 2;
        if (val >= t[0]) return 3;
        return 4;
      } else if (statKey === 'REB') {
        // Higher is slightly good, but never full green
        if (val >= t[3]) return 3;
        if (val >= t[2]) return 3;
        if (val >= t[1]) return 2;
        if (val >= t[0]) return 2;
        return 2;
      } else if (statKey === 'BLK' || statKey === 'STL') {
        // Higher is better (greener)
        if (val < t[0]) return 0;
        if (val < t[1]) return 1;
        if (val < t[2]) return 2;
        if (val < t[3]) return 3;
        return 4;
      }
    }
    // fallback
    return 2;
  };

  const rating = RATING_SCALE[getRatingIndex(stat, value)];
  const safeValue = typeof value === 'number' && !isNaN(value) ? value : 0;

  return (
    <div className="statbar-group">
      {title && <div className="statbar-title label-micro">{title}</div>}
      <div className={`statbar-row statbar-${rating.key}`}>
        <span className="statbar-name">{STAT_LABELS[stat] || stat}</span>
        <div className="statbar-track">
          <div className="statbar-fill" style={{ width: `${rating.fill}%` }} />
        </div>
        <span className="statbar-rating">{rating.label}</span>
        <span className="statbar-value num">{safeValue.toFixed(1)}</span>
      </div>
    </div>
  );
};

export default StatMiniBar;
