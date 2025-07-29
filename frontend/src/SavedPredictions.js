import React, { useState, useEffect } from 'react';
import SavedPredictionCard from './SavedPredictionCard';
import './SavedPredictions.css';

const SavedPredictions = () => {
  const [savedPredictions, setSavedPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSavedPredictions();
  }, []);

  const loadSavedPredictions = () => {
    try {
      const saved = localStorage.getItem('savedPredictions');
      if (saved) {
        setSavedPredictions(JSON.parse(saved));
      }
    } catch (error) {
      console.error('Error loading saved predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteSavedPrediction = (id) => {
    try {
      const updatedPredictions = savedPredictions.filter(pred => pred.id !== id);
      setSavedPredictions(updatedPredictions);
      localStorage.setItem('savedPredictions', JSON.stringify(updatedPredictions));
    } catch (error) {
      console.error('Error deleting saved prediction:', error);
    }
  };

  const clearAllPredictions = () => {
    if (window.confirm('Are you sure you want to delete all saved predictions?')) {
      try {
        setSavedPredictions([]);
        localStorage.removeItem('savedPredictions');
      } catch (error) {
        console.error('Error clearing saved predictions:', error);
      }
    }
  };

  if (loading) {
    return (
      <div className="saved-predictions-container">
        <div className="saved-predictions-header">
          <h2 className="saved-predictions-title">Saved Predictions</h2>
        </div>
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading saved predictions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="saved-predictions-container">
      <div className="saved-predictions-header">
        <h2 className="saved-predictions-title">Saved Predictions</h2>
        {savedPredictions.length > 0 && (
          <button 
            className="clear-all-button"
            onClick={clearAllPredictions}
          >
            Clear All
          </button>
        )}
      </div>
      
      {savedPredictions.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No Saved Predictions</h3>
          <p>Your saved predictions will appear here. Run a prediction on the home page and click "Save" to get started!</p>
        </div>
      ) : (
        <div className="saved-predictions-grid">
          {savedPredictions.map((savedPrediction, index) => (
            <SavedPredictionCard
              key={savedPrediction.id}
              savedPrediction={savedPrediction}
              onDelete={deleteSavedPrediction}
              index={index}
              total={savedPredictions.length}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default SavedPredictions;