# PlayerAveragesChart Component

A React component that displays player statistical averages as a bar chart using Chart.js.

## Props

- `category` (string): The statistical category being displayed (e.g., "Points", "Rebounds", "Points+Rebounds+Assists")
- `seasonAverage` (number): The player's season average for the category
- `recentAverage` (number): The player's average over the last 10 games
- `h2hAverage` (number): The player's head-to-head average against the opponent
- `opponentAbbr` (string): The opponent team abbreviation (e.g., "LAL", "BOS")

## Features

- Responsive bar chart with three bars: Season, Last 10 Games, and vs. Opponent
- Color-coded bars for easy distinction
- Handles combined categories (e.g., "Points+Rebounds+Assists")
- Fallback text display in case chart rendering fails
- Input validation to handle edge cases
- Styled to match the app's dark theme

## Usage

```jsx
import PlayerAveragesChart from './PlayerAveragesChart';

<PlayerAveragesChart
  category="Points"
  seasonAverage={25.5}
  recentAverage={24.2}
  h2hAverage={26.1}
  opponentAbbr="LAL"
/>
```

## Dependencies

- chart.js
- react-chartjs-2
- React

## Styling

The component uses the `.chart-container` CSS class for styling, which provides:
- Semi-transparent background
- Rounded corners
- Box shadow
- Border styling 
