import React from 'react';
import { render } from '@testing-library/react';
import PlayerAveragesChart from './PlayerAveragesChart';

// Only the rendering layer is mocked. chart.js itself stays real -- the
// annotation/datalabels plugins subclass its internals at import time, so a
// hand-rolled chart.js mock is far more brittle than simply never mounting a
// canvas. Mocking `react-chartjs-2` is what keeps jsdom out of canvas code.
jest.mock('react-chartjs-2', () => ({
  Bar: () => <div data-testid="bar-chart">Mock Chart</div>,
}));

jest.mock('chartjs-plugin-datalabels', () => ({ id: 'datalabels' }));
jest.mock('chartjs-plugin-annotation', () => ({ id: 'annotation' }));

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
