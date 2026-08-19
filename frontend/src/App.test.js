import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import App from './App';

jest.mock('react-chartjs-2', () => ({
  Bar: () => <div data-testid="bar-chart">Mock Chart</div>,
}));
jest.mock('chartjs-plugin-datalabels', () => ({ id: 'datalabels' }));
jest.mock('chartjs-plugin-annotation', () => ({ id: 'annotation' }));

const PLAYERS = [
  { id: 201939, full_name: 'Stephen Curry' },
  { id: 2544, full_name: 'LeBron James' },
];
const TEAMS = [
  { id: 1610612747, abbreviation: 'LAL', full_name: 'Los Angeles Lakers' },
];

beforeEach(() => {
  localStorage.clear();
  global.fetch = jest.fn((url) => {
    if (url.includes('/api/all-players')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(PLAYERS) });
    }
    if (url.includes('/api/teams')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(TEAMS) });
    }
    if (url.includes('/api/player-details/')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ team_abbreviation: 'GSW', position: 'Guard', jersey: '30' }),
      });
    }
    return Promise.resolve({ ok: false, status: 404, text: () => Promise.resolve('nope') });
  });
});

afterEach(() => {
  jest.resetAllMocks();
});

test('renders the markets page and loads reference data', async () => {
  render(<App />);

  expect(screen.getByRole('heading', { name: /player prop markets/i })).toBeInTheDocument();
  expect(screen.getByText(/no market open/i)).toBeInTheDocument();

  // Player and team counts appear in the header once the fetches resolve.
  await waitFor(() => {
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/all-players'),
      expect.anything()
    );
  });
});

test('selecting a player from search opens the market ticket', async () => {
  render(<App />);

  // Wait for the player list to land before typing -- the dropdown filters
  // over loaded state, so an early keystroke would match nothing.
  await screen.findByText('2', { selector: '.header-stat-value' });

  const search = screen.getByLabelText(/search for an nba player/i);
  fireEvent.focus(search);
  fireEvent.change(search, { target: { value: 'Curry' } });

  const option = await screen.findByText('Stephen Curry');
  fireEvent.mouseDown(option);

  // The ticket replaces the "no market open" empty state.
  await waitFor(() => {
    expect(screen.queryByText(/no market open/i)).not.toBeInTheDocument();
  });
  expect(screen.getByRole('heading', { name: 'Stephen Curry' })).toBeInTheDocument();
  expect(screen.getByLabelText(/^market$/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /price this market/i })).toBeDisabled();
});

test('shows an error banner when the backend is unreachable', async () => {
  global.fetch = jest.fn(() => Promise.reject(new Error('Failed to fetch')));
  render(<App />);

  const alert = await screen.findByRole('alert', {}, { timeout: 8000 });
  expect(within(alert).getByText(/could not reach the prediction backend/i)).toBeInTheDocument();
}, 15000);

test('positions tab shows the empty state with no saved predictions', async () => {
  render(<App />);

  fireEvent.click(screen.getByRole('tab', { name: /positions/i }));

  expect(await screen.findByRole('heading', { name: /^positions$/i })).toBeInTheDocument();
  expect(screen.getByText(/no positions yet/i)).toBeInTheDocument();
});
