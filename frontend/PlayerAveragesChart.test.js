import React from 'react';
import { render } from '@testing-library/react';
import PlayerAveragesChart from './PlayerAveragesChart';

// Mock Chart.js to avoid canvas issues in tests
jest.mock('chart.js', () => ({
  Chart: {
    register: jest.fn(),
  },
  CategoryScale: jest.fn(),
  LinearScale: jest.fn(),
  BarElement: jest.fn(),
  Title: jest.fn(),
  Tooltip: jest.fn(),
  Legend: jest.fn(),
}));

jest.mock('react-chartjs-2', () => ({
  Bar: () => <div data-testid="bar-chart">Mock Chart</div>,
}));

describe('PlayerAveragesChart', () => {
  const defaultProps = {
    category: 'Points',
    seasonAverage: 25.5,
    recentAverage: 24.2,
    h2hAverage: 26.1,
    opponentAbbr: 'LAL',
  };

  it('renders without crashing', () => {
    const { getByTestId } = render(<PlayerAveragesChart {...defaultProps} />);
    expect(getByTestId('bar-chart')).toBeInTheDocument();
  });

  it('renders with different category', () => {
    const { getByTestId } = render(
      <PlayerAveragesChart {...defaultProps} category="Rebounds" />
    );
    expect(getByTestId('bar-chart')).toBeInTheDocument();
  });

  it('renders with combined category', () => {
    const { getByTestId } = render(
      <PlayerAveragesChart {...defaultProps} category="Points+Rebounds+Assists" />
    );
    expect(getByTestId('bar-chart')).toBeInTheDocument();
  });
}); 