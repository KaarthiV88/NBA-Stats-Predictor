import React from 'react';

const StatMiniBar = ({ stat, value, category }) => {
  // Updated NBA-contextual thresholds for each stat
  const thresholds = {
    'PTS': [109, 112, 115, 118],
    'REB': [40, 41.9, 43.5, 45.9],
    'AST': [20, 21.9, 25.5, 25.9],
    'BLK': [4.0, 4.4, 5.1, 5.4],
    'STL': [7.0, 7.4, 7.9, 8.4]
  };
  // Color and fill mapping (with new yellow-green and vibrant green)
  const colorFillMap = [
    { color: 'red', fill: 20, style: { backgroundColor: '#ef4444' } },
    { color: 'orange', fill: 40, style: { backgroundColor: '#f97316' } },
    { color: 'yellow', fill: 60, style: { backgroundColor: '#fbbf24' } },
    { color: 'yellow-green', fill: 80, style: { backgroundColor: '#CCFF00' } },
    { color: 'vibrant-green', fill: 100, style: { backgroundColor: '#22c55e' } }
  ];

  // Determine stat context (offensive/defensive)
  const offensiveStats = ['Points', 'Assists', 'Rebounds', 'Points+Rebounds+Assists', 'Rebounds+Assists', 'Points+Rebounds', 'Points+Assists'];
  const defensiveStats = ['Blocks', 'Steals', 'Blocks+Steals'];
  const isOffensive = offensiveStats.some(cat => category && category.includes(cat));
  const isDefensive = defensiveStats.some(cat => category && category.includes(cat));

  // Get color/fill index based on value and context
  const getColorFillIndex = (stat, value) => {
    const t = thresholds[stat];
    if (!t) return 2; // default to yellow

    // --- OFFENSIVE CATEGORY LOGIC ---
    if (isOffensive) {
      if (stat === 'PTS' || stat === 'AST') {
        // Higher is better (greener)
        if (value < t[0]) return 0;
        if (value < t[1]) return 1;
        if (value < t[2]) return 2;
        if (value < t[3]) return 3;
        return 4;
      } else if (stat === 'REB' || stat === 'BLK' || stat === 'STL') {
        // Higher is worse (redder)
        if (value >= t[3]) return 0;
        if (value >= t[2]) return 1;
        if (value >= t[1]) return 2;
        if (value >= t[0]) return 3;
        return 4;
      }
    }
    // --- DEFENSIVE CATEGORY LOGIC ---
    if (isDefensive) {
      if (stat === 'PTS' || stat === 'AST') {
        // Higher is worse (redder)
        if (value >= t[3]) return 0;
        if (value >= t[2]) return 1;
        if (value >= t[1]) return 2;
        if (value >= t[0]) return 3;
        return 4;
      } else if (stat === 'REB') {
        // Higher is slightly good (yellow-green, but never full green)
        if (value >= t[3]) return 3; // yellow-green
        if (value >= t[2]) return 3; // yellow-green
        if (value >= t[1]) return 2; // yellow
        if (value >= t[0]) return 2; // yellow
        return 2; // always yellow or yellow-green
      } else if (stat === 'BLK' || stat === 'STL') {
        // Higher is better (greener)
        if (value < t[0]) return 0;
        if (value < t[1]) return 1;
        if (value < t[2]) return 2;
        if (value < t[3]) return 3;
        return 4;
      }
    }
    // fallback
    return 2;
  };

  const colorFillIndex = getColorFillIndex(stat, value);
  const { color, fill, style } = colorFillMap[colorFillIndex];
  const fillPercentage = fill;

  return (
    <div className="flex items-center space-x-2 mb-2">
      <span className="text-sm font-medium text-gray-700 min-w-[80px]">
        {stat === 'PTS' ? 'Points' :
         stat === 'REB' ? 'Rebounds' :
         stat === 'AST' ? 'Assists' :
         stat === 'BLK' ? 'Blocks' :
         stat === 'STL' ? 'Steals' : stat}:
      </span>
      <div className="flex-1 bg-gray-200 rounded-full h-2">
        <div
          className="h-2 rounded-full transition-all duration-300"
          style={{
            width: `${fillPercentage}%`,
            ...style
          }}
        ></div>
      </div>
      <span className="text-sm font-semibold text-gray-800 min-w-[40px] text-right">
        {value.toFixed(1)}
      </span>
    </div>
  );
};

export default StatMiniBar; 
