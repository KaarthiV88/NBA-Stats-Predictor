import React from 'react';
import './Navbar.css';

const Navbar = ({ currentPage, onPageChange }) => {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-links">
          <button
            className={`navbar-link ${currentPage === 'home' ? 'active' : ''}`}
            onClick={() => onPageChange('home')}
          >
            Home
          </button>
          <button
            className={`navbar-link ${currentPage === 'saved' ? 'active' : ''}`}
            onClick={() => onPageChange('saved')}
          >
            Saved Predictions
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 