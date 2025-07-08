import React, { useState, useEffect, useRef } from 'react';
import './App.css';

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
            ref={playerInputRef}
          />
          {showPlayerDropdown && searchTerm && (
            <ul className="dropdown">
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
            <div className={`player-card ${initialShiftApplied ? 'shifted' : ''}`}>
              <div className="player-image-container">
                <img
                  src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${selectedPlayer.id}.png?imwidth=1040&imheight=760`}
                  alt={selectedPlayer.full_name}
                  className="player-image"
                  onError={(e) => { e.target.src = 'https://via.placeholder.com/128'; }}
                />
              </div>
              <h2 className="player-name">{selectedPlayer.full_name}</h2>
              <div className="input-group">
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="select-input"
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
                />
              </div>
              <div className="search-container">
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
                  className="search-input"
                  ref={opponentInputRef}
                />
                {showOpponentDropdown && opponentSearch && (
                  <ul className="dropdown">
                    {filteredTeams.length > 0 ? (
                      filteredTeams.map((team, index) => (
                        <li
                          key={team.id}
                          onClick={() => {
                            setSelectedOpponent(team);
                            setOpponentSearch(team.full_name);
                            setShowOpponentDropdown(false);
                            setOpponentDropdownIndex(-1);
                          }}
                          className={`dropdown-item ${index === opponentDropdownIndex ? 'selected' : ''}`}
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
              <select
                value={seasonType}
                onChange={(e) => setSeasonType(e.target.value)}
                className="select-input"
              >
                <option value="Regular Season">Regular Season</option>
                <option value="Playoffs">Playoffs</option>
              </select>
              <button onClick={handleRunPrediction} className="predict-button">Run Prediction</button>
            </div>
          )}
          {(selectedPlayer && (loading || prediction)) && (
            <div className={`prediction-card ${initialShiftApplied ? 'shifted' : ''}`}>
              {loading ? (
                <div className="loading-overlay">
                  <div className="progress-bar-container">
                    <div className="progress-bar" style={{ width: `${progress}%` }}></div>
                  </div>
                  <div className="loading-text">{progress}%</div>
                </div>
              ) : (
                prediction && (
                  <>
                    <div className="prediction-result">
                      <h3 className="prediction-title">Prediction Result</h3>
                    </div>
                    <div className="h2h-table">
                      <h4>Head-to-Head Matchups vs. {selectedOpponent?.abbreviation || 'Opponent'}:</h4>
                      {prediction.h2h_list && prediction.h2h_list.length > 0 ? (
                        <table>
                          <thead>
                            <tr>
                              <th>Date</th>
                              <th>Matchup</th>
                              <th>PTS</th>
                              <th>REB</th>
                              <th>AST</th>
                              <th>BLK</th>
                              <th>STL</th>
                            </tr>
                          </thead>
                          <tbody>
                            {prediction.h2h_list.map((game, index) => (
                              <tr key={index}>
                                <td>{game.Game_Date}</td>
                                <td>{game.Matchup}</td>
                                <td>{game.PTS.toFixed(1)}</td>
                                <td>{game.REB.toFixed(1)}</td>
                                <td>{game.AST.toFixed(1)}</td>
                                <td>{game.BLK.toFixed(1)}</td>
                                <td>{game.STL.toFixed(1)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <p>No head-to-head data available</p>
                      )}
                    </div>
                    <div className="prediction-outcome">
                      <h4 className="prediction-title">Player Averages for {category}:</h4>
                      <div className="prediction-value">
                        <p>Season: {prediction.player_averages.season_averages[category === 'Points' ? 'PTS' : category === 'Rebounds' ? 'REB' : category === 'Assists' ? 'AST' : category === 'Blocks' ? 'BLK' : category === 'Steals' ? 'STL' : 'PTS'].toFixed(1)}</p>
                        <p>Last 10 Games: {prediction.player_averages.recent_averages[category === 'Points' ? 'PTS' : category === 'Rebounds' ? 'REB' : category === 'Assists' ? 'AST' : category === 'Blocks' ? 'BLK' : category === 'Steals' ? 'STL' : 'PTS'].toFixed(1)}</p>
                        <p>vs. {selectedOpponent?.abbreviation || 'Opponent'}: {prediction.h2h_list.reduce((sum, game) => sum + (category === 'Points' ? game.PTS : category === 'Rebounds' ? game.REB : category === 'Assists' ? game.AST : category === 'Blocks' ? game.BLK : category === 'Steals' ? game.STL : 0), 0) / (prediction.h2h_list.length || 1).toFixed(1)}</p>
                      </div>
                    </div>
                    <div className="prediction-outcome">
                      <h4 className="prediction-title">Opponent Defensive Averages (Last 10 Games):</h4>
                      <div className="prediction-value">
                        <p>Points: 113.8</p>
                        <p>Rebounds: 49.2</p>
                        <p>Assists: 28.3</p>
                        <p>Blocks: 5.2</p>
                        <p>Steals: 5.8</p>
                        <p>Defensive Rating: 110.0</p>
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
                )
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
