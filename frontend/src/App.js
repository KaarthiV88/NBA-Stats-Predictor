import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import PlayerAveragesChart from './PlayerAveragesChart';
import StatMiniBar from './StatMiniBar';

// Betting categories (matched with backend)
const bettingCategories = [
  'Points', 'Rebounds', 'Assists', 'Blocks', 'Steals',
  'Points+Rebounds+Assists', 'Rebounds+Assists',
  'Points+Rebounds', 'Points+Assists', 'Blocks+Steals'
];

// Fallback player list for testing
const fallbackPlayers = [
  { id: 2544, full_name: "LeBron James" },
  { id: 201939, full_name: "Stephen Curry" },
  { id: 203076, full_name: "Kevin Durant" }
];

function App() {
  const [players, setPlayers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [category, setCategory] = useState('');
  const [bettingLine, setBettingLine] = useState('');
  const [opponentSearch, setOpponentSearch] = useState('');
  const [selectedOpponent, setSelectedOpponent] = useState(null);
  const [seasonType, setSeasonType] = useState('Regular Season');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [showPlayerDropdown, setShowPlayerDropdown] = useState(false);
  const [showOpponentDropdown, setShowOpponentDropdown] = useState(false);
  const [loading, setLoading] = useState(false);
  const [playerDropdownIndex, setPlayerDropdownIndex] = useState(-1);
  const [opponentDropdownIndex, setOpponentDropdownIndex] = useState(-1);
  const [initialShiftApplied, setInitialShiftApplied] = useState(false);
  const playerInputRef = useRef(null);
  const opponentInputRef = useRef(null);
  const [progress, setProgress] = useState(0);

  const fetchPlayerDetails = async (playerName) => {
    try {
      console.log('Fetching details for:', playerName);
      const encodedName = encodeURIComponent(playerName);
      console.log('Encoded name:', encodedName);
      const response = await fetch(`http://localhost:5001/api/player-details/${encodedName}`, { mode: 'cors' });
      console.log('Response status:', response.status);
      if (response.ok) {
        const data = await response.json();
        console.log('Fetched player details:', data);
        return {
          height: data.height,
          weight: data.weight,
          jersey: data.jersey,
          position: data.position,
          team_name: data.team_name,
          team_city: data.team_city,
          team_abbreviation: data.team_abbreviation,
          team_color: data.team_color,
          school: data.school,
          country: data.country,
          season_exp: data.season_exp
        };
      } else {
        const errorText = await response.text();
        console.error('API error:', response.status, errorText);
      }
    } catch (error) {
      console.error('Error fetching player details:', error);
    }
    return null;
  };

  useEffect(() => {
    const fetchData = async (retryCount = 3) => {
      setLoading(true);
      for (let attempt = 1; attempt <= retryCount; attempt++) {
        try {
          const playersResponse = await fetch('http://localhost:5001/api/all-players', { mode: 'cors', signal: AbortSignal.timeout(20000) });
          if (!playersResponse.ok) throw new Error(`HTTP ${playersResponse.status}: ${await playersResponse.text()}`);
          const playersData = await playersResponse.json();
          if (!Array.isArray(playersData)) throw new Error('Invalid players data format');
          setPlayers(playersData.sort((a, b) => a.full_name.localeCompare(b.full_name)));

          const teamsResponse = await fetch('http://localhost:5001/api/teams', { mode: 'cors', signal: AbortSignal.timeout(20000) });
          if (!teamsResponse.ok) throw new Error(`HTTP ${teamsResponse.status}: ${await teamsResponse.text()}`);
          const teamsData = await teamsResponse.json();
          if (!Array.isArray(teamsData)) throw new Error('Invalid teams data format');
          setTeams(teamsData.sort((a, b) => a.full_name.localeCompare(b.full_name)));

          setError(null);
          break;
        } catch (err) {
          console.error(`Attempt ${attempt} failed:`, err);
          if (attempt === retryCount) {
            console.warn('Falling back to default players due to fetch failure');
            setPlayers(fallbackPlayers);
            setError(`Error fetching data: ${err.message}. Check backend at http://localhost:5001 or console logs.`);
          } else {
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
          }
        }
      }
      setLoading(false);
    };
    fetchData();
  }, []);

  // Fetch player details when selectedPlayer changes
  useEffect(() => {
    if (selectedPlayer && selectedPlayer.full_name && !selectedPlayer.height) {
      console.log('Fetching details for selected player:', selectedPlayer.full_name);
      fetchPlayerDetails(selectedPlayer.full_name).then(details => {
        console.log('Details fetched:', details);
        if (details) {
          setSelectedPlayer(prev => {
            const updated = { ...prev, ...details };
            console.log('Updated player with details:', updated);
            return updated;
          });
        }
      });
    }
  }, [selectedPlayer?.full_name]);

  const filteredPlayers = players.filter(player =>
    player.full_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredTeams = teams.filter(team =>
    team.full_name.toLowerCase().includes(opponentSearch.toLowerCase()) ||
    team.abbreviation.toLowerCase().includes(opponentSearch.toLowerCase())
  );

  const handleRunPrediction = async () => {
    if (!selectedPlayer || !category || !bettingLine || !selectedOpponent) {
      setError('Please fill in all fields.');
      return;
    }
    setLoading(true);
    setProgress(0);
    setPrediction(null);
    setInitialShiftApplied(true); // Trigger shift when prediction starts

    let interval;
    try {
      interval = setInterval(() => {
        setProgress((prev) => (prev >= 100 ? 100 : prev + 20));
      }, 500);

      const response = await fetch(
        `http://localhost:5001/api/predict?player_name=${encodeURIComponent(selectedPlayer.full_name)}` +
        `&category=${encodeURIComponent(category)}` +
        `&opponent_abbr=${encodeURIComponent(selectedOpponent.abbreviation)}` +
        `&betting_line=${encodeURIComponent(bettingLine)}` +
        `&season_type=${encodeURIComponent(seasonType)}`,
        { mode: 'cors', signal: AbortSignal.timeout(20000) }
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      const data = await response.json();
      console.log('Prediction data:', data);
      setPrediction(data);
      setError(null);
    } catch (err) {
      console.error('Prediction fetch failed:', err);
      setError(`Error fetching prediction: ${err.message}`);
      setPrediction(null);
    } finally {
      setLoading(false);
      if (interval) clearInterval(interval);
      setProgress(100);
    }
  };

  const handleRetryFetch = () => {
    setError(null);
    setPlayers([]);
    setTeams([]);
    setLoading(true);
    const fetchData = async () => {
      try {
        const playersResponse = await fetch('http://localhost:5001/api/all-players', { mode: 'cors', signal: AbortSignal.timeout(20000) });
        if (!playersResponse.ok) throw new Error(`HTTP ${playersResponse.status}: ${await playersResponse.text()}`);
        const playersData = await playersResponse.json();
        if (!Array.isArray(playersData)) throw new Error('Invalid players data format');
        setPlayers(playersData.sort((a, b) => a.full_name.localeCompare(b.full_name)));

        const teamsResponse = await fetch('http://localhost:5001/api/teams', { mode: 'cors', signal: AbortSignal.timeout(20000) });
        if (!teamsResponse.ok) throw new Error(`HTTP ${teamsResponse.status}: ${await teamsResponse.text()}`);
        const teamsData = await teamsResponse.json();
        if (!Array.isArray(teamsData)) throw new Error('Invalid teams data format');
        setTeams(teamsData.sort((a, b) => a.full_name.localeCompare(b.full_name)));

        setError(null);
      } catch (err) {
        console.error('Retry failed:', err);
        setError(`Retry failed: ${err.message}. Check backend at http://localhost:5001.`);
        setPlayers(fallbackPlayers);
      }
      setLoading(false);
    };
    fetchData();
  };

  const handlePlayerKeyDown = (e) => {
    if (!showPlayerDropdown || !filteredPlayers.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setPlayerDropdownIndex((prev) => Math.min(prev + 1, filteredPlayers.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setPlayerDropdownIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && playerDropdownIndex >= 0) {
      e.preventDefault();
      const selected = filteredPlayers[playerDropdownIndex];
      setSelectedPlayer(selected);
      setSearchTerm(selected.full_name);
      setShowPlayerDropdown(false);
      setPlayerDropdownIndex(-1);
      setPrediction(null);
      setInitialShiftApplied(false); // Reset shift when player changes
    }
  };

  const handleOpponentKeyDown = (e) => {
    if (!showOpponentDropdown || !filteredTeams.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setOpponentDropdownIndex((prev) => Math.min(prev + 1, filteredTeams.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setOpponentDropdownIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && opponentDropdownIndex >= 0) {
      e.preventDefault();
      const selected = filteredTeams[opponentDropdownIndex];
      setSelectedOpponent(selected);
      setOpponentSearch(selected.full_name);
      setShowOpponentDropdown(false);
      setOpponentDropdownIndex(-1);
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">NBA Stats Predictor</h1>
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={handleRetryFetch} className="retry-button">Retry</button>
        </div>
      )}
      <div className="content-wrapper">
        <div className="search-container">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setShowPlayerDropdown(true);
              setPlayerDropdownIndex(-1);
            }}
            onFocus={() => setShowPlayerDropdown(true)}
            onBlur={() => setTimeout(() => setShowPlayerDropdown(false), 200)}
            onKeyDown={handlePlayerKeyDown}
            placeholder="Search for an NBA player..."
            className="search-input"
            style={{ borderColor: '#3EB489' }} // App signature color
            ref={playerInputRef}
          />
          {showPlayerDropdown && searchTerm && (
            <ul className="dropdown" style={{ borderColor: '#3EB489' }}>
              {filteredPlayers.length > 0 ? (
                filteredPlayers.map((player, index) => (
                  <li
                    key={player.id}
                    onClick={() => {
                      setSelectedPlayer(player);
                      setSearchTerm(player.full_name);
                      setShowPlayerDropdown(false);
                      setPlayerDropdownIndex(-1);
                      setPrediction(null);
                      setInitialShiftApplied(false); // Reset shift when player changes
                    }}
                    className={`dropdown-item ${index === playerDropdownIndex ? 'selected' : ''}`}
                    style={{ backgroundColor: index === playerDropdownIndex ? '#3EB489' : 'transparent' }}
                  >
                    {player.full_name}
                  </li>
                ))
              ) : (
                <li className="dropdown-item no-results">No players found</li>
              )}
            </ul>
          )}
        </div>

        <div className="cards-container">
          {selectedPlayer && (
            <div className={`player-card ${initialShiftApplied ? 'shifted' : ''}`} 
                 style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>
              <div className="player-info-container">
                <div className="player-image-container">
                  <img
                    src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${selectedPlayer.id}.png?imwidth=1040&imheight=760`}
                    alt={selectedPlayer.full_name}
                    className="player-image"
                    style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}
                    onError={(e) => { e.target.src = 'https://via.placeholder.com/128'; }}
                  />
                </div>
                <div className="player-details">
                  <h2 className="player-name" style={{ color: selectedPlayer.team_color || '#3EB489' }}>
                    {selectedPlayer.full_name}
                  </h2>
                  <div className="player-info-grid">
                    <div className="player-info-item">
                      <span className="player-info-label">Height:</span>
                      <span className="player-info-value">{selectedPlayer.height || 'N/A'}</span>
                    </div>
                    <div className="player-info-item">
                      <span className="player-info-label">Weight:</span>
                      <span className="player-info-value">{selectedPlayer.weight ? `${selectedPlayer.weight} lbs` : 'N/A'}</span>
                    </div>
                    <div className="player-info-item">
                      <span className="player-info-label">Jersey:</span>
                      <span className="player-info-value">#{selectedPlayer.jersey || 'N/A'}</span>
                    </div>
                    <div className="player-info-item">
                      <span className="player-info-label">Position:</span>
                      <span className="player-info-value">{selectedPlayer.position || 'N/A'}</span>
                    </div>
                    <div className="player-info-item">
                      <span className="player-info-label">Team:</span>
                      <span className="player-info-value">{selectedPlayer.team_city && selectedPlayer.team_name ? `${selectedPlayer.team_city} ${selectedPlayer.team_name}` : 'N/A'}</span>
                    </div>
                    <div className="player-info-item">
                      <span className="player-info-label">School:</span>
                      <span className="player-info-value">{selectedPlayer.school || 'N/A'}</span>
                    </div>
                    <div className="player-info-item">
                      <span className="player-info-label">Country:</span>
                      <span className="player-info-value">{selectedPlayer.country || 'N/A'}</span>
                    </div>
                    <div className="player-info-item">
                      <span className="player-info-label">Experience:</span>
                      <span className="player-info-value">{selectedPlayer.season_exp ? `${selectedPlayer.season_exp} years` : 'N/A'}</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="input-group">
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="select-input"
                  style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}
                >
                  <option value="" disabled>Select Betting Category</option>
                  {bettingCategories.map(cat => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
                <input
                  type="number"
                  value={bettingLine}
                  onChange={(e) => setBettingLine(e.target.value)}
                  placeholder="Betting Line"
                  className="text-input"
                  style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}
                />
              </div>
              <div className="input-group">
                <div className="search-container" style={{ position: 'relative', width: '100%' }}>
                  <input
                    type="text"
                    value={opponentSearch}
                    onChange={(e) => {
                      setOpponentSearch(e.target.value);
                      setShowOpponentDropdown(true);
                      setOpponentDropdownIndex(-1);
                    }}
                    onFocus={() => setShowOpponentDropdown(true)}
                    onBlur={() => setTimeout(() => setShowOpponentDropdown(false), 200)}
                    onKeyDown={handleOpponentKeyDown}
                    placeholder="Search for opponent team..."
                    className="text-input"
                    ref={opponentInputRef}
                    style={{ borderColor: selectedPlayer.team_color || '#3EB489', width: '100%' }}
                  />
                  {showOpponentDropdown && opponentSearch && (
                    <ul className="dropdown" style={{ 
                      borderColor: selectedPlayer.team_color || '#3EB489',
                      position: 'absolute',
                      top: '100%',
                      left: 0,
                      right: 0,
                      zIndex: 1000
                    }}>
                      {filteredTeams.length > 0 ? (
                        filteredTeams.map((team, index) => (
                          <li
                            key={team.id}
                            onClick={() => {
                              setSelectedOpponent(team);
                              setOpponentSearch(team.full_name);
                              setShowOpponentDropdown(false);
                              setOpponentDropdownIndex(-1);
                              setPrediction(null);
                              setInitialShiftApplied(false);
                            }}
                            className={`dropdown-item ${index === opponentDropdownIndex ? 'selected' : ''}`}
                            style={{ backgroundColor: index === opponentDropdownIndex ? (selectedPlayer.team_color || '#3EB489') : 'transparent' }}
                            onMouseEnter={e => e.currentTarget.style.backgroundColor = selectedPlayer.team_color || '#3EB489'}
                            onMouseLeave={e => e.currentTarget.style.backgroundColor = index === opponentDropdownIndex ? (selectedPlayer.team_color || '#3EB489') : 'transparent'}
                          >
                            {team.full_name} ({team.abbreviation})
                          </li>
                        ))
                      ) : (
                        <li className="dropdown-item no-results">No teams found</li>
                      )}
                    </ul>
                  )}
                </div>
              </div>
              <div className="input-group">
                <select
                  value={seasonType}
                  onChange={(e) => setSeasonType(e.target.value)}
                  className="select-input"
                  style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}
                >
                  <option value="Regular Season">Regular Season</option>
                  <option value="Playoffs">Playoffs</option>
                </select>
              </div>
              <button 
                onClick={handleRunPrediction} 
                className="predict-button"
                style={{ 
                  backgroundColor: selectedPlayer.team_color || '#3EB489',
                  color: '#181A20'
                }}
              >
                Run Prediction
              </button>
            </div>
          )}

          {(selectedPlayer && (loading || prediction)) && (
            <div className={`prediction-card ${initialShiftApplied ? 'shifted' : ''}`}
                 style={{ 
                   backgroundColor: '#23242A', // Keep dark gray background
                   borderColor: selectedPlayer.team_color || '#3EB489',
                   boxShadow: `0 0 20px ${selectedPlayer.team_color || '#3EB489'}50`
                 }}>
              {loading ? (
                <div className="loading-overlay">
                  <div className="progress-bar-container">
                    <div className="progress-bar" style={{ 
                      width: `${progress}%`,
                      backgroundColor: selectedPlayer.team_color || '#3EB489' 
                    }}></div>
                  </div>
                  <div className="loading-text">{progress}%</div>
                </div>
              ) : prediction ? (
                <>
                  <div className="prediction-result">
                    <h3 className="prediction-title">Prediction Result</h3>
                  </div>
                  <div className="h2h-table">
                    <h4>Head-to-Head Matchups vs. {selectedOpponent?.abbreviation || 'Opponent'}:</h4>
                    {prediction.h2h_list && prediction.h2h_list.length > 0 ? (
                      <table style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>
                        <thead>
                          <tr>
                            <th style={{ backgroundColor: selectedPlayer.team_color || '#3EB489', borderColor: selectedPlayer.team_color || '#3EB489' }}>Date</th>
                            <th style={{ backgroundColor: selectedPlayer.team_color || '#3EB489', borderColor: selectedPlayer.team_color || '#3EB489' }}>Matchup</th>
                            <th style={{ backgroundColor: selectedPlayer.team_color || '#3EB489', borderColor: selectedPlayer.team_color || '#3EB489' }}>PTS</th>
                            <th style={{ backgroundColor: selectedPlayer.team_color || '#3EB489', borderColor: selectedPlayer.team_color || '#3EB489' }}>REB</th>
                            <th style={{ backgroundColor: selectedPlayer.team_color || '#3EB489', borderColor: selectedPlayer.team_color || '#3EB489' }}>AST</th>
                            <th style={{ backgroundColor: selectedPlayer.team_color || '#3EB489', borderColor: selectedPlayer.team_color || '#3EB489' }}>BLK</th>
                            <th style={{ backgroundColor: selectedPlayer.team_color || '#3EB489', borderColor: selectedPlayer.team_color || '#3EB489' }}>STL</th>
                          </tr>
                        </thead>
                        <tbody>
                          {prediction.h2h_list.map((game, index) => (
                            <tr key={index}>
                              <td style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>{game.Game_Date}</td>
                              <td style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>{game.Matchup}</td>
                              <td style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>{game.PTS.toFixed(1)}</td>
                              <td style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>{game.REB.toFixed(1)}</td>
                              <td style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>{game.AST.toFixed(1)}</td>
                              <td style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>{game.BLK.toFixed(1)}</td>
                              <td style={{ borderColor: selectedPlayer.team_color || '#3EB489' }}>{game.STL.toFixed(1)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p>No head-to-head data available</p>
                    )}
                  </div>
                  <div className="prediction-outcome">
                    <PlayerAveragesChart
                      category={category}
                      seasonAverage={(() => {
                        if (category === 'Points+Rebounds+Assists') {
                          return (prediction.player_averages.season_averages.PTS || 0) + 
                                 (prediction.player_averages.season_averages.REB || 0) + 
                                 (prediction.player_averages.season_averages.AST || 0);
                        } else if (category === 'Rebounds+Assists') {
                          return (prediction.player_averages.season_averages.REB || 0) + 
                                 (prediction.player_averages.season_averages.AST || 0);
                        } else if (category === 'Points+Rebounds') {
                          return (prediction.player_averages.season_averages.PTS || 0) + 
                                 (prediction.player_averages.season_averages.REB || 0);
                        } else if (category === 'Points+Assists') {
                          return (prediction.player_averages.season_averages.PTS || 0) + 
                                 (prediction.player_averages.season_averages.AST || 0);
                        } else if (category === 'Blocks+Steals') {
                          return (prediction.player_averages.season_averages.BLK || 0) + 
                                 (prediction.player_averages.season_averages.STL || 0);
                        } else {
                          return prediction.player_averages.season_averages[category === 'Points' ? 'PTS' : category === 'Rebounds' ? 'REB' : category === 'Assists' ? 'AST' : category === 'Blocks' ? 'BLK' : category === 'Steals' ? 'STL' : 'PTS'] || 0;
                        }
                      })()}
                      recentAverage={(() => {
                        if (category === 'Points+Rebounds+Assists') {
                          return (prediction.player_averages.recent_averages.PTS || 0) + 
                                 (prediction.player_averages.recent_averages.REB || 0) + 
                                 (prediction.player_averages.recent_averages.AST || 0);
                        } else if (category === 'Rebounds+Assists') {
                          return (prediction.player_averages.recent_averages.REB || 0) + 
                                 (prediction.player_averages.recent_averages.AST || 0);
                        } else if (category === 'Points+Rebounds') {
                          return (prediction.player_averages.recent_averages.PTS || 0) + 
                                 (prediction.player_averages.recent_averages.REB || 0);
                        } else if (category === 'Points+Assists') {
                          return (prediction.player_averages.recent_averages.PTS || 0) + 
                                 (prediction.player_averages.recent_averages.AST || 0);
                        } else if (category === 'Blocks+Steals') {
                          return (prediction.player_averages.recent_averages.BLK || 0) + 
                                 (prediction.player_averages.recent_averages.STL || 0);
                        } else {
                          return prediction.player_averages.recent_averages[category === 'Points' ? 'PTS' : category === 'Rebounds' ? 'REB' : category === 'Assists' ? 'AST' : category === 'Blocks' ? 'BLK' : category === 'Steals' ? 'STL' : 'PTS'] || 0;
                        }
                      })()}
                      h2hAverage={prediction.h2h_list && prediction.h2h_list.length > 0 ? 
                        prediction.h2h_list.reduce((sum, game) => {
                          let gameValue = 0;
                          if (category === 'Points+Rebounds+Assists') {
                            gameValue = game.PTS + game.REB + game.AST;
                          } else if (category === 'Rebounds+Assists') {
                            gameValue = game.REB + game.AST;
                          } else if (category === 'Points+Rebounds') {
                            gameValue = game.PTS + game.REB;
                          } else if (category === 'Points+Assists') {
                            gameValue = game.PTS + game.AST;
                          } else if (category === 'Blocks+Steals') {
                            gameValue = game.BLK + game.STL;
                          } else {
                            gameValue = category === 'Points' ? game.PTS : category === 'Rebounds' ? game.REB : category === 'Assists' ? game.AST : category === 'Blocks' ? game.BLK : category === 'Steals' ? game.STL : 0;
                          }
                          return sum + gameValue;
                        }, 0) / prediction.h2h_list.length : 0}
                      opponentAbbr={selectedOpponent?.abbreviation || 'Opponent'}
                      bettingLine={parseFloat(bettingLine)}
                      teamColor={selectedPlayer.team_color || '#3EB489'}
                    />
                  </div>
                  <div className="prediction-outcome">
                    <h4 className="prediction-title">Opponent Defensive Averages (Last 10 Games):</h4>
                    <div className="prediction-value">
                      {(() => {
                        const isDefensiveCategory = ['Blocks', 'Steals', 'Blocks+Steals'].includes(category);
                        return (
                          <div className="stats-mini-bars">
                            <StatMiniBar 
                              stat="PTS" 
                              value={prediction.opp_averages?.PTS ?? 0} 
                              category={category}
                              isDefensiveCategory={isDefensiveCategory}
                            />
                            <StatMiniBar 
                              stat="REB" 
                              value={prediction.opp_averages?.REB ?? 0} 
                              category={category}
                              isDefensiveCategory={isDefensiveCategory}
                            />
                            <StatMiniBar 
                              stat="AST" 
                              value={prediction.opp_averages?.AST ?? 0} 
                              category={category}
                              isDefensiveCategory={isDefensiveCategory}
                            />
                            <StatMiniBar 
                              stat="BLK" 
                              value={prediction.opp_averages?.BLK ?? 0} 
                              category={category}
                              isDefensiveCategory={isDefensiveCategory}
                            />
                            <StatMiniBar 
                              stat="STL" 
                              value={prediction.opp_averages?.STL ?? 0} 
                              category={category}
                              isDefensiveCategory={isDefensiveCategory}
                            />
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                  <div className="prediction-outcome">
                    <h4 className="prediction-title">Predicted Outcome:</h4>
                    <div className="prediction-value">
                      {prediction.message && (
                        <>
                          <p>Predicted {category}: {prediction.message.match(/Predicted [A-Za-z]+: (\d+\.\d+)/)?.[1] || 'N/A'} (95% CI: {prediction.message.match(/95% CI: (\d+\.\d+-\d+\.\d+)/)?.[1] || 'N/A'})</p>
                          <p>P(Over {bettingLine}): {prediction.message.match(/P\(Over \d+\.\d+\): (\d+\.\d+)%/)?.[1] || 'N/A'}%</p>
                          <p className="bet-recommendation">{prediction.message.match(/(\d+\.\d+)% confident bet on [A-Z]+/)?.[1] || 'N/A'}% confident bet on {prediction.bet_on?.toUpperCase() || 'N/A'}</p>
                        </>
                      )}
                    </div>
                  </div>
                </>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
