import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import PlayerAveragesChart from './PlayerAveragesChart';
import StatMiniBar from './StatMiniBar';
import PredictionScorecard from './PredictionScorecard';
import Navbar from './Navbar';
import SavedPredictions from './SavedPredictions';
import { apiUrl } from './api';

// Betting categories (matched with backend)
const bettingCategories = [
  'Points', 'Rebounds', 'Assists', 'Blocks', 'Steals',
  'Points+Rebounds+Assists', 'Rebounds+Assists',
  'Points+Rebounds', 'Points+Assists', 'Blocks+Steals'
];

// Which raw box-score keys make up each betting category. Combined categories
// are just the sum of their parts, so one table drives every lookup.
const CATEGORY_STATS = {
  'Points': ['PTS'],
  'Rebounds': ['REB'],
  'Assists': ['AST'],
  'Blocks': ['BLK'],
  'Steals': ['STL'],
  'Points+Rebounds+Assists': ['PTS', 'REB', 'AST'],
  'Rebounds+Assists': ['REB', 'AST'],
  'Points+Rebounds': ['PTS', 'REB'],
  'Points+Assists': ['PTS', 'AST'],
  'Blocks+Steals': ['BLK', 'STL']
};

const sumCategory = (source, category) =>
  (CATEGORY_STATS[category] || ['PTS']).reduce((total, key) => total + (source?.[key] || 0), 0);

// Short ticker-style label for a category, e.g. "PRA"
const CATEGORY_TICKER = {
  'Points': 'PTS',
  'Rebounds': 'REB',
  'Assists': 'AST',
  'Blocks': 'BLK',
  'Steals': 'STL',
  'Points+Rebounds+Assists': 'PRA',
  'Rebounds+Assists': 'RA',
  'Points+Rebounds': 'PR',
  'Points+Assists': 'PA',
  'Blocks+Steals': 'BS'
};

// Fallback player list for testing
const fallbackPlayers = [
  { id: 2544, full_name: "LeBron James" },
  { id: 201939, full_name: "Stephen Curry" },
  { id: 203076, full_name: "Kevin Durant" }
];

const readSavedCount = () => {
  try {
    const saved = localStorage.getItem('savedPredictions');
    return saved ? JSON.parse(saved).length : 0;
  } catch {
    return 0;
  }
};

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
  const playerInputRef = useRef(null);
  const opponentInputRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const [currentPage, setCurrentPage] = useState('home');
  const [savedCount, setSavedCount] = useState(readSavedCount);
  const [toast, setToast] = useState(null);

  const showToast = useCallback((message, tone = 'success') => {
    setToast({ message, tone });
    setTimeout(() => setToast(null), 3200);
  }, []);

  const fetchPlayerDetails = async (playerName) => {
    try {
      const encodedName = encodeURIComponent(playerName);
      const response = await fetch(apiUrl(`/api/player-details/${encodedName}`), { mode: 'cors' });
      if (response.ok) {
        const data = await response.json();
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
      }
      console.error('API error:', response.status, await response.text());
    } catch (err) {
      console.error('Error fetching player details:', err);
    }
    return null;
  };

  // Shared loader for the player + team reference lists.
  const loadReferenceData = useCallback(async (retryCount = 3) => {
    setLoading(true);
    for (let attempt = 1; attempt <= retryCount; attempt++) {
      try {
        const playersResponse = await fetch(apiUrl('/api/all-players'), { mode: 'cors', signal: AbortSignal.timeout(20000) });
        if (!playersResponse.ok) throw new Error(`HTTP ${playersResponse.status}: ${await playersResponse.text()}`);
        const playersData = await playersResponse.json();
        if (!Array.isArray(playersData)) throw new Error('Invalid players data format');
        setPlayers(playersData.sort((a, b) => a.full_name.localeCompare(b.full_name)));

        const teamsResponse = await fetch(apiUrl('/api/teams'), { mode: 'cors', signal: AbortSignal.timeout(20000) });
        if (!teamsResponse.ok) throw new Error(`HTTP ${teamsResponse.status}: ${await teamsResponse.text()}`);
        const teamsData = await teamsResponse.json();
        if (!Array.isArray(teamsData)) throw new Error('Invalid teams data format');
        setTeams(teamsData.sort((a, b) => a.full_name.localeCompare(b.full_name)));

        setError(null);
        setLoading(false);
        return;
      } catch (err) {
        console.error(`Attempt ${attempt} failed:`, err);
        if (attempt === retryCount) {
          console.warn('Falling back to default players due to fetch failure');
          setPlayers(fallbackPlayers);
          setError(`Could not reach the prediction backend: ${err.message}`);
        } else {
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
      }
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadReferenceData();
  }, [loadReferenceData]);

  // Fetch player details when selectedPlayer changes
  useEffect(() => {
    if (selectedPlayer && selectedPlayer.full_name && !selectedPlayer.height) {
      fetchPlayerDetails(selectedPlayer.full_name).then(details => {
        if (details) {
          setSelectedPlayer(prev => ({ ...prev, ...details }));
        }
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPlayer?.full_name]);

  const filteredPlayers = players.filter(player =>
    player.full_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredTeams = teams.filter(team =>
    team.full_name.toLowerCase().includes(opponentSearch.toLowerCase()) ||
    team.abbreviation.toLowerCase().includes(opponentSearch.toLowerCase())
  );

  const isTicketReady = Boolean(selectedPlayer && category && bettingLine && selectedOpponent);

  const handleRunPrediction = async () => {
    if (!isTicketReady) {
      setError('Select a player, category, line and opponent before pricing the market.');
      return;
    }
    setLoading(true);
    setProgress(0);
    setPrediction(null);

    let interval;
    try {
      interval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? 90 : prev + 5));
      }, 1000);

      const response = await fetch(
        apiUrl(`/api/predict?player_name=${encodeURIComponent(selectedPlayer.full_name)}`) +
        `&category=${encodeURIComponent(category)}` +
        `&opponent_abbr=${encodeURIComponent(selectedOpponent.abbreviation)}` +
        `&betting_line=${encodeURIComponent(bettingLine)}` +
        `&season_type=${encodeURIComponent(seasonType)}`,
        { mode: 'cors', signal: AbortSignal.timeout(120000) }
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      const data = await response.json();
      setPrediction(data);
      setError(null);
    } catch (err) {
      console.error('Prediction fetch failed:', err);
      if (err.name === 'TimeoutError') {
        setError('Pricing timed out. The model is taking longer than expected — try again, or check that the backend is running.');
      } else if (err.name === 'AbortError') {
        setError('Request was cancelled. Please try again.');
      } else {
        setError(`Error fetching prediction: ${err.message}`);
      }
      setPrediction(null);
    } finally {
      setLoading(false);
      if (interval) clearInterval(interval);
      setProgress(100);
    }
  };

  const handleSavePrediction = () => {
    if (!prediction || !selectedPlayer || !selectedOpponent) return;

    try {
      const savedPrediction = {
        id: Date.now().toString(),
        player: selectedPlayer,
        category,
        bettingLine,
        opponent: selectedOpponent,
        seasonType,
        prediction,
        savedAt: new Date().toISOString()
      };

      const existingSaved = localStorage.getItem('savedPredictions');
      const savedPredictions = existingSaved ? JSON.parse(existingSaved) : [];
      savedPredictions.unshift(savedPrediction);
      localStorage.setItem('savedPredictions', JSON.stringify(savedPredictions));
      setSavedCount(savedPredictions.length);
      showToast('Position added');
    } catch (err) {
      console.error('Error saving prediction:', err);
      showToast('Could not save position', 'error');
    }
  };

  const handleRetryFetch = () => {
    setError(null);
    setPlayers([]);
    setTeams([]);
    loadReferenceData(1);
  };

  const selectPlayer = (player) => {
    setSelectedPlayer(player);
    setSearchTerm(player.full_name);
    setShowPlayerDropdown(false);
    setPlayerDropdownIndex(-1);
    setPrediction(null);
  };

  const selectOpponent = (team) => {
    setSelectedOpponent(team);
    setOpponentSearch(team.full_name);
    setShowOpponentDropdown(false);
    setOpponentDropdownIndex(-1);
    setPrediction(null);
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
      selectPlayer(filteredPlayers[playerDropdownIndex]);
    } else if (e.key === 'Escape') {
      setShowPlayerDropdown(false);
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
      selectOpponent(filteredTeams[opponentDropdownIndex]);
    } else if (e.key === 'Escape') {
      setShowOpponentDropdown(false);
    }
  };

  const accent = selectedPlayer?.team_color || '#3EB489';
  const ticker = CATEGORY_TICKER[category] || '';

  const h2hAverage = prediction?.h2h_list?.length
    ? prediction.h2h_list.reduce((sum, game) => sum + sumCategory(game, category), 0) / prediction.h2h_list.length
    : 0;

  return (
    <div className="app-container">
      <Navbar currentPage={currentPage} onPageChange={setCurrentPage} positionCount={savedCount} />

      {toast && (
        <div className={`toast toast-${toast.tone}`} role="status">{toast.message}</div>
      )}

      {currentPage === 'saved' ? (
        <SavedPredictions onCountChange={setSavedCount} />
      ) : (
        <main className="page">
          <header className="page-header">
            <div>
              <h1 className="page-title">Player Prop Markets</h1>
              <p className="page-subtitle">
                Price any NBA player prop against the book. The model trains on the
                2024-25 and 2025-26 seasons and quotes both sides in cents.
              </p>
            </div>
            <div className="page-header-stats">
              <div className="header-stat">
                <span className="label-micro">Players</span>
                <span className="header-stat-value num">{players.length || '--'}</span>
              </div>
              <div className="header-stat">
                <span className="label-micro">Teams</span>
                <span className="header-stat-value num">{teams.length || '--'}</span>
              </div>
              <div className="header-stat">
                <span className="label-micro">Positions</span>
                <span className="header-stat-value num">{savedCount}</span>
              </div>
            </div>
          </header>

          {error && (
            <div className="banner banner-error" role="alert">
              <span className="banner-icon" aria-hidden="true">!</span>
              <p className="banner-text">{error}</p>
              <button onClick={handleRetryFetch} className="btn btn-ghost">Retry</button>
            </div>
          )}

          {/* --- Market search ------------------------------------------- */}
          <div className="search-block">
            <div className="search-container">
              <span className="search-icon" aria-hidden="true">⌕</span>
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
                placeholder="Search a player to open their market…"
                className="search-input"
                aria-label="Search for an NBA player"
                ref={playerInputRef}
              />
              {showPlayerDropdown && searchTerm && (
                <ul className="dropdown">
                  {filteredPlayers.length > 0 ? (
                    filteredPlayers.slice(0, 60).map((player, index) => (
                      <li
                        key={player.id}
                        onMouseDown={() => selectPlayer(player)}
                        onMouseEnter={() => setPlayerDropdownIndex(index)}
                        className={`dropdown-item ${index === playerDropdownIndex ? 'selected' : ''}`}
                      >
                        <span>{player.full_name}</span>
                        <span className="dropdown-hint label-micro">Open</span>
                      </li>
                    ))
                  ) : (
                    <li className="dropdown-item no-results">No players found</li>
                  )}
                </ul>
              )}
            </div>
          </div>

          {!selectedPlayer ? (
            <div className="empty-state">
              <div className="empty-icon" aria-hidden="true">◎</div>
              <h3>No market open</h3>
              <p>Search for a player above to build a prop market and have the model price it.</p>
            </div>
          ) : (
            <div className="market-layout">
              {/* --- Left: the ticket ---------------------------------- */}
              <section className="ticket market-surface" style={{ '--accent-team': accent }}>
                <div className="ticket-player">
                  <div className="ticket-portrait">
                    <img
                      src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${selectedPlayer.id}.png?imwidth=1040&imheight=760`}
                      alt={selectedPlayer.full_name}
                      className="ticket-avatar"
                      onError={(e) => { e.target.style.visibility = 'hidden'; }}
                    />
                  </div>
                  <div className="ticket-identity">
                    <h2 className="ticket-name">{selectedPlayer.full_name}</h2>
                    <div className="ticket-meta">
                      <span className="chip chip-accent">
                        {selectedPlayer.team_abbreviation || '—'}
                      </span>
                      <span className="chip">#{selectedPlayer.jersey || '—'}</span>
                      <span className="chip">{selectedPlayer.position || '—'}</span>
                    </div>
                  </div>
                </div>

                <dl className="ticket-vitals">
                  {[
                    ['Height', selectedPlayer.height],
                    ['Weight', selectedPlayer.weight ? `${selectedPlayer.weight} lb` : null],
                    ['Team', selectedPlayer.team_name],
                    ['Exp', selectedPlayer.season_exp != null ? `${selectedPlayer.season_exp} yr` : null],
                    ['School', selectedPlayer.school],
                    ['Country', selectedPlayer.country]
                  ].map(([label, value]) => (
                    <div className="vital" key={label}>
                      <dt className="label-micro">{label}</dt>
                      <dd className="vital-value">{value || '—'}</dd>
                    </div>
                  ))}
                </dl>

                <hr className="hairline" />

                <div className="ticket-form">
                  <div className="field">
                    <label className="label-micro" htmlFor="category">Market</label>
                    <select
                      id="category"
                      value={category}
                      onChange={(e) => { setCategory(e.target.value); setPrediction(null); }}
                      className="input"
                    >
                      <option value="" disabled>Select a stat category</option>
                      {bettingCategories.map(cat => (
                        <option key={cat} value={cat}>{cat}</option>
                      ))}
                    </select>
                  </div>

                  <div className="field">
                    <label className="label-micro" htmlFor="line">Line</label>
                    <input
                      id="line"
                      type="number"
                      step="0.5"
                      value={bettingLine}
                      onChange={(e) => { setBettingLine(e.target.value); setPrediction(null); }}
                      placeholder="25.5"
                      className="input num"
                    />
                  </div>

                  <div className="field field-full">
                    <label className="label-micro" htmlFor="opponent">Opponent</label>
                    <div className="search-container">
                      <input
                        id="opponent"
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
                        placeholder="Search team…"
                        className="input"
                        ref={opponentInputRef}
                      />
                      {showOpponentDropdown && opponentSearch && (
                        <ul className="dropdown">
                          {filteredTeams.length > 0 ? (
                            filteredTeams.map((team, index) => (
                              <li
                                key={team.id}
                                onMouseDown={() => selectOpponent(team)}
                                onMouseEnter={() => setOpponentDropdownIndex(index)}
                                className={`dropdown-item ${index === opponentDropdownIndex ? 'selected' : ''}`}
                              >
                                <span>{team.full_name}</span>
                                <span className="dropdown-hint num">{team.abbreviation}</span>
                              </li>
                            ))
                          ) : (
                            <li className="dropdown-item no-results">No teams found</li>
                          )}
                        </ul>
                      )}
                    </div>
                  </div>

                  <div className="field field-full">
                    <label className="label-micro" htmlFor="seasonType">Season type</label>
                    <div className="segmented" role="group" id="seasonType">
                      {['Regular Season', 'Playoffs'].map(opt => (
                        <button
                          key={opt}
                          type="button"
                          className={`segmented-option ${seasonType === opt ? 'active' : ''}`}
                          onClick={() => { setSeasonType(opt); setPrediction(null); }}
                        >
                          {opt}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {isTicketReady && (
                  <div className="ticket-summary">
                    <span className="ticket-summary-ticker num">{ticker}</span>
                    <span className="ticket-summary-text">
                      {selectedPlayer.full_name} <strong className="num">{bettingLine}</strong>{' '}
                      {category} vs {selectedOpponent.abbreviation}
                    </span>
                  </div>
                )}

                <button
                  onClick={handleRunPrediction}
                  className="btn btn-primary btn-block"
                  disabled={loading || !isTicketReady}
                >
                  {loading ? 'Pricing market…' : 'Price this market'}
                </button>
              </section>

              {/* --- Right: the priced market -------------------------- */}
              <section className="market-result">
                {loading ? (
                  <div className="pricing-panel market-surface">
                    <span className="label-micro">Pricing</span>
                    <div className="pricing-headline">Running the model</div>
                    <p className="pricing-note">
                      Fetching game logs, building features and training the ensemble.
                      This usually takes 30–90 seconds.
                    </p>
                    <div className="progress-bar-container">
                      <div className="progress-bar" style={{ width: `${progress}%` }} />
                    </div>
                    <div className="pricing-percent num">{progress}%</div>
                  </div>
                ) : prediction ? (
                  <>
                    <PredictionScorecard
                      prediction={prediction}
                      category={category}
                      bettingLine={bettingLine}
                      teamColor={accent}
                    />

                    <div className="panel market-surface">
                      <div className="panel-head">
                        <h3 className="panel-title">Form</h3>
                        <span className="label-micro">Season · Last 10 · H2H</span>
                      </div>
                      <PlayerAveragesChart
                        category={category}
                        seasonAverage={sumCategory(prediction.player_averages?.season_averages, category)}
                        recentAverage={sumCategory(prediction.player_averages?.recent_averages, category)}
                        h2hAverage={h2hAverage}
                        opponentAbbr={selectedOpponent?.abbreviation || 'OPP'}
                        bettingLine={parseFloat(bettingLine)}
                        teamColor={accent}
                      />
                    </div>

                    <div className="panel market-surface">
                      <div className="panel-head">
                        <h3 className="panel-title">
                          Matchup vs {selectedOpponent?.abbreviation || 'OPP'}
                        </h3>
                        <span className="label-micro">Opponent last 10</span>
                      </div>
                      <div className="statbars">
                        {['PTS', 'REB', 'AST', 'BLK', 'STL'].map(stat => (
                          <StatMiniBar
                            key={stat}
                            stat={stat}
                            value={prediction.opp_averages?.[stat] ?? 0}
                            category={category}
                          />
                        ))}
                      </div>
                    </div>

                    <div className="panel market-surface">
                      <div className="panel-head">
                        <h3 className="panel-title">
                          Head-to-head vs {selectedOpponent?.abbreviation || 'OPP'}
                        </h3>
                        <span className="label-micro">
                          {prediction.h2h_list?.length || 0} games
                        </span>
                      </div>
                      {prediction.h2h_list && prediction.h2h_list.length > 0 ? (
                        <div className="table-scroll">
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>Date</th>
                                <th>Matchup</th>
                                <th className="ta-right">PTS</th>
                                <th className="ta-right">REB</th>
                                <th className="ta-right">AST</th>
                                <th className="ta-right">BLK</th>
                                <th className="ta-right">STL</th>
                                <th className="ta-right">{ticker || 'TOT'}</th>
                              </tr>
                            </thead>
                            <tbody>
                              {prediction.h2h_list.map((game, index) => {
                                const total = sumCategory(game, category);
                                const line = parseFloat(bettingLine);
                                const hit = Number.isFinite(line) ? total > line : null;
                                return (
                                  <tr key={index}>
                                    <td>{game.Game_Date}</td>
                                    <td>{game.Matchup}</td>
                                    <td className="ta-right num">{game.PTS.toFixed(0)}</td>
                                    <td className="ta-right num">{game.REB.toFixed(0)}</td>
                                    <td className="ta-right num">{game.AST.toFixed(0)}</td>
                                    <td className="ta-right num">{game.BLK.toFixed(0)}</td>
                                    <td className="ta-right num">{game.STL.toFixed(0)}</td>
                                    <td className={`ta-right num cell-total ${hit === null ? '' : hit ? 'over' : 'under'}`}>
                                      {total.toFixed(1)}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="panel-empty">No head-to-head data available.</p>
                      )}
                    </div>

                    <button
                      onClick={handleSavePrediction}
                      className="btn btn-secondary btn-block"
                    >
                      Add to positions
                    </button>
                  </>
                ) : (
                  <div className="empty-state empty-state-inline">
                    <div className="empty-icon" aria-hidden="true">⌁</div>
                    <h3>Market not priced</h3>
                    <p>
                      {isTicketReady
                        ? 'Ready to go — hit “Price this market”.'
                        : 'Choose a market, line and opponent to enable pricing.'}
                    </p>
                  </div>
                )}
              </section>
            </div>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
