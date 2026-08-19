import React, { useState, useEffect, useCallback, useMemo } from 'react';
import SavedPredictionCard from './SavedPredictionCard';
import './SavedPredictions.css';

const SORTS = {
  recent: { label: 'Newest', compare: (a, b) => new Date(b.savedAt) - new Date(a.savedAt) },
  confidence: {
    label: 'Confidence',
    compare: (a, b) => (b.prediction?.confidence || 0) - (a.prediction?.confidence || 0)
  },
  player: {
    label: 'Player',
    compare: (a, b) => (a.player?.full_name || '').localeCompare(b.player?.full_name || '')
  }
};

const SavedPredictions = ({ onCountChange }) => {
  const [savedPredictions, setSavedPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState('recent');
  const [sideFilter, setSideFilter] = useState('all');

  const syncCount = useCallback((list) => {
    if (onCountChange) onCountChange(list.length);
  }, [onCountChange]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem('savedPredictions');
      const parsed = saved ? JSON.parse(saved) : [];
      setSavedPredictions(parsed);
      syncCount(parsed);
    } catch (error) {
      console.error('Error loading saved predictions:', error);
    } finally {
      setLoading(false);
    }
  }, [syncCount]);

  const persist = (list) => {
    setSavedPredictions(list);
    localStorage.setItem('savedPredictions', JSON.stringify(list));
    syncCount(list);
  };

  const deleteSavedPrediction = (id) => {
    try {
      persist(savedPredictions.filter(pred => pred.id !== id));
    } catch (error) {
      console.error('Error deleting saved prediction:', error);
    }
  };

  const clearAllPredictions = () => {
    if (window.confirm('Delete all saved positions? This cannot be undone.')) {
      try {
        setSavedPredictions([]);
        localStorage.removeItem('savedPredictions');
        syncCount([]);
      } catch (error) {
        console.error('Error clearing saved predictions:', error);
      }
    }
  };

  // Portfolio-level summary across every saved position.
  const summary = useMemo(() => {
    if (!savedPredictions.length) return null;
    const overs = savedPredictions.filter(p => p.prediction?.bet_on === 'over').length;
    const confidences = savedPredictions
      .map(p => p.prediction?.confidence)
      .filter(c => typeof c === 'number' && !isNaN(c));
    const avg = confidences.length
      ? confidences.reduce((a, b) => a + b, 0) / confidences.length
      : null;
    const strong = confidences.filter(c => c >= 75).length;
    return { total: savedPredictions.length, overs, unders: savedPredictions.length - overs, avg, strong };
  }, [savedPredictions]);

  const visible = useMemo(() => {
    const filtered = sideFilter === 'all'
      ? savedPredictions
      : savedPredictions.filter(p => p.prediction?.bet_on === sideFilter);
    return [...filtered].sort(SORTS[sortKey].compare);
  }, [savedPredictions, sideFilter, sortKey]);

  if (loading) {
    return (
      <main className="page">
        <div className="loading-container">
          <div className="loading-spinner" />
          <p>Loading positions…</p>
        </div>
      </main>
    );
  }

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <h1 className="page-title">Positions</h1>
          <p className="page-subtitle">
            Markets you've priced and saved. Stored locally in this browser.
          </p>
        </div>
        {savedPredictions.length > 0 && (
          <button className="btn btn-ghost btn-danger" onClick={clearAllPredictions}>
            Clear all
          </button>
        )}
      </header>

      {summary && (
        <div className="portfolio-summary">
          <div className="summary-stat">
            <span className="label-micro">Open positions</span>
            <span className="summary-value num">{summary.total}</span>
          </div>
          <div className="summary-stat">
            <span className="label-micro">Over / Under</span>
            <span className="summary-value num">
              <span className="over">{summary.overs}</span>
              <span className="summary-slash">/</span>
              <span className="under">{summary.unders}</span>
            </span>
          </div>
          <div className="summary-stat">
            <span className="label-micro">Avg confidence</span>
            <span className="summary-value num">
              {summary.avg === null ? '--' : `${summary.avg.toFixed(1)}%`}
            </span>
          </div>
          <div className="summary-stat">
            <span className="label-micro">Strong (75%+)</span>
            <span className="summary-value num">{summary.strong}</span>
          </div>
        </div>
      )}

      {savedPredictions.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon" aria-hidden="true">◎</div>
          <h3>No positions yet</h3>
          <p>Price a market on the Markets tab and add it here to track it.</p>
        </div>
      ) : (
        <>
          <div className="positions-toolbar">
            <div className="segmented segmented-inline" role="group" aria-label="Filter by side">
              {[['all', 'All'], ['over', 'Over'], ['under', 'Under']].map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  className={`segmented-option ${sideFilter === key ? 'active' : ''}`}
                  onClick={() => setSideFilter(key)}
                >
                  {label}
                </button>
              ))}
            </div>

            <label className="sort-control">
              <span className="label-micro">Sort</span>
              <select
                className="input"
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value)}
              >
                {Object.entries(SORTS).map(([key, { label }]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </label>
          </div>

          {visible.length === 0 ? (
            <div className="empty-state empty-state-inline">
              <div className="empty-icon" aria-hidden="true">⌁</div>
              <h3>No {sideFilter} positions</h3>
              <p>Nothing matches this filter.</p>
            </div>
          ) : (
            <div className="positions-grid">
              {visible.map((savedPrediction) => (
                <SavedPredictionCard
                  key={savedPrediction.id}
                  savedPrediction={savedPrediction}
                  onDelete={deleteSavedPrediction}
                />
              ))}
            </div>
          )}
        </>
      )}
    </main>
  );
};

export default SavedPredictions;
