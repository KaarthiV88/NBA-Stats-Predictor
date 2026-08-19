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
          <img
            className="navbar-mark"
            src={`${process.env.PUBLIC_URL}/kv_money_market.jpeg`}
            alt="KV Money Market"
          />
          <span className="navbar-wordmark">
            KV <span className="navbar-wordmark-accent">Money Market</span>
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
