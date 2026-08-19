import React from 'react';
import './Navbar.css';

const NAV_ITEMS = [
  { key: 'home', label: 'Markets' },
  { key: 'saved', label: 'Positions' }
];

const Navbar = ({ currentPage, onPageChange, positionCount = 0 }) => {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <span className="navbar-mark" aria-hidden="true" />
          <span className="navbar-wordmark">
            NBA<span className="navbar-wordmark-accent">Predictor</span>
          </span>
        </div>

        <div className="navbar-links" role="tablist" aria-label="Sections">
          {NAV_ITEMS.map(({ key, label }) => (
            <button
              key={key}
              role="tab"
              aria-selected={currentPage === key}
              className={`navbar-link ${currentPage === key ? 'active' : ''}`}
              onClick={() => onPageChange(key)}
            >
              {label}
              {key === 'saved' && positionCount > 0 && (
                <span className="navbar-badge num">{positionCount}</span>
              )}
            </button>
          ))}
        </div>

        <div className="navbar-status">
          <span className="navbar-live-dot" aria-hidden="true" />
          <span className="navbar-status-text">Model live</span>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
