import React from 'react';
import './PredictionScorecard.css';

/**
 * Renders a model prediction the way a prediction market renders a contract:
 * two opposing sides (OVER / UNDER) each quoted as an implied price in cents,
 * with the model's edge over the posted line called out explicitly.
 */
const PredictionScorecard = ({ prediction, category, bettingLine, teamColor = '#3EB489', compact = false }) => {
  if (!prediction) {
    return null;
  }

  const { bet_on: betOn, confidence, predicted_value: predictedValue, confidence_interval: confidenceInterval } = prediction;

  const lineNum = parseFloat(bettingLine);
  const predictedNum = typeof predictedValue === 'number' ? predictedValue : parseFloat(predictedValue);
  const hasPrediction = Number.isFinite(predictedNum);
  const hasLine = Number.isFinite(lineNum);

  // Model confidence is the implied probability of the side the model favours.
  // The opposing side is its complement, so the two always price to 100c --
  // exactly how a binary market quotes YES/NO.
  const rawConfidence = Number.isFinite(confidence) ? Math.min(Math.max(confidence, 0), 100) : null;
  const favorsOver = betOn === 'over';
  const overPrice = rawConfidence === null ? null : (favorsOver ? rawConfidence : 100 - rawConfidence);
  const underPrice = rawConfidence === null ? null : 100 - overPrice;

  const formatPrice = (p) => (p === null ? '--' : `${Math.round(p)}¢`);

  const difference = hasPrediction && hasLine ? predictedNum - lineNum : null;
  const differenceFormatted =
    difference === null ? '--' : `${difference > 0 ? '+' : ''}${difference.toFixed(1)}`;

  // Edge is what a bettor actually cares about: how far the model sits from the
  // posted line, as a share of that line.
  const edgePct =
    difference !== null && hasLine && lineNum !== 0
      ? (difference / lineNum) * 100
      : null;

  const sideLabel = betOn ? betOn.toUpperCase() : 'NO SIGNAL';
  const sideClass = favorsOver ? 'over' : 'under';

  // Confidence bands mirror how markets describe conviction.
  const conviction =
    rawConfidence === null ? null
      : rawConfidence >= 75 ? 'Strong'
        : rawConfidence >= 62 ? 'Moderate'
          : rawConfidence >= 54 ? 'Slight'
            : 'Toss-up';

  return (
    <div className={`scorecard ${compact ? 'scorecard-compact' : ''}`}>
      {/* --- Market question ------------------------------------------- */}
      <div className="scorecard-question">
        <span className="label-micro">Contract</span>
        <div className="scorecard-question-text">
          {category} <span className="scorecard-question-line num">{hasLine ? lineNum : '--'}</span>
        </div>
      </div>

      {/* --- The two sides, priced -------------------------------------- */}
      <div className="scorecard-book">
        <div className={`scorecard-side scorecard-side-over ${favorsOver ? 'is-favored' : ''}`}>
          <div className="scorecard-side-head">
            <span className="scorecard-side-name">Over</span>
            {favorsOver && <span className="scorecard-side-tag">Model pick</span>}
          </div>
          <div className="scorecard-side-price num">{formatPrice(overPrice)}</div>
        </div>

        <div className={`scorecard-side scorecard-side-under ${!favorsOver ? 'is-favored' : ''}`}>
          <div className="scorecard-side-head">
            <span className="scorecard-side-name">Under</span>
            {!favorsOver && <span className="scorecard-side-tag">Model pick</span>}
          </div>
          <div className="scorecard-side-price num">{formatPrice(underPrice)}</div>
        </div>
      </div>

      {/* --- Split probability bar -------------------------------------- */}
      <div
        className="scorecard-probbar"
        role="img"
        aria-label={`Implied probability: over ${formatPrice(overPrice)}, under ${formatPrice(underPrice)}`}
      >
        <div className="scorecard-probbar-over" style={{ width: `${overPrice ?? 50}%` }} />
        <div className="scorecard-probbar-under" style={{ width: `${underPrice ?? 50}%` }} />
      </div>
      <div className="scorecard-probbar-legend">
        <span className="scorecard-legend-over num">{formatPrice(overPrice)} over</span>
        {conviction && <span className="scorecard-conviction">{conviction} conviction</span>}
        <span className="scorecard-legend-under num">{formatPrice(underPrice)} under</span>
      </div>

      {/* --- Model numbers ---------------------------------------------- */}
      <div className="scorecard-stats">
        <div className="scorecard-stat">
          <span className="label-micro">Model projection</span>
          <span className="scorecard-stat-value num" style={{ color: teamColor }}>
            {hasPrediction ? predictedNum.toFixed(1) : '--'}
          </span>
          {confidenceInterval && (
            <span className="scorecard-stat-sub num">95% CI {confidenceInterval}</span>
          )}
        </div>

        <div className="scorecard-stat">
          <span className="label-micro">Posted line</span>
          <span className="scorecard-stat-value num">{hasLine ? lineNum.toFixed(1) : '--'}</span>
          <span className="scorecard-stat-sub">Sportsbook</span>
        </div>

        <div className="scorecard-stat">
          <span className="label-micro">Edge</span>
          <span className={`scorecard-stat-value num ${difference !== null && difference >= 0 ? 'positive' : 'negative'}`}>
            {differenceFormatted}
          </span>
          <span className="scorecard-stat-sub num">
            {edgePct === null ? '--' : `${edgePct > 0 ? '+' : ''}${edgePct.toFixed(1)}%`}
          </span>
        </div>
      </div>

      {/* --- Plain-language read ---------------------------------------- */}
      {hasPrediction && hasLine && (
        <p className="scorecard-readout">
          Model projects <strong className="num">{predictedNum.toFixed(1)}</strong>{' '}
          {category.toLowerCase()},{' '}
          <strong className={difference >= 0 ? 'positive' : 'negative'}>
            <span className="num">{differenceFormatted}</span>
          </strong>{' '}
          vs the {lineNum} line &mdash; implying{' '}
          <strong className={sideClass}>
            {sideLabel} at <span className="num">{formatPrice(favorsOver ? overPrice : underPrice)}</span>
          </strong>
          .
        </p>
      )}
    </div>
  );
};

export default PredictionScorecard;
