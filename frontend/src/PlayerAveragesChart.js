import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartDataLabels,
  annotationPlugin
);

const PlayerAveragesChart = ({ category, seasonAverage, recentAverage, h2hAverage, opponentAbbr, bettingLine, teamColor = '#3EB489' }) => {
  const validSeasonAverage = typeof seasonAverage === 'number' && !isNaN(seasonAverage) ? seasonAverage : 0;
  const validRecentAverage = typeof recentAverage === 'number' && !isNaN(recentAverage) ? recentAverage : 0;
  const validH2hAverage = typeof h2hAverage === 'number' && !isNaN(h2hAverage) ? h2hAverage : 0;
  const validBettingLine = typeof bettingLine === 'number' && !isNaN(bettingLine) ? bettingLine : null;

  // Create dynamic bar colors based on team color
  const createBarColors = (teamColor) => {
    // Convert hex to RGB for manipulation
    const hex = teamColor.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    
    // Create lighter and darker variations
    const lighten = (color, factor) => {
      const newR = Math.min(255, Math.round(r + (255 - r) * factor));
      const newG = Math.min(255, Math.round(g + (255 - g) * factor));
      const newB = Math.min(255, Math.round(b + (255 - b) * factor));
      return `rgba(${newR}, ${newG}, ${newB}, 0.6)`;
    };
    
    const darken = (color, factor) => {
      const newR = Math.max(0, Math.round(r * (1 - factor)));
      const newG = Math.max(0, Math.round(g * (1 - factor)));
      const newB = Math.max(0, Math.round(b * (1 - factor)));
      return `rgba(${newR}, ${newG}, ${newB}, 0.85)`;
    };
    
    return [
      lighten(teamColor, 0.3), // Lightest
      `rgba(${r}, ${g}, ${b}, 0.7)`, // Medium
      darken(teamColor, 0.2) // Darkest
    ];
  };

  const barColors = createBarColors(teamColor);
  const borderColors = barColors.map(color => color.replace(/[^,]+\)/, '1)'));

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: `Player Averages for ${category}`,
        color: '#fff',
        font: {
          size: 22,
          weight: 'bold',
        },
        padding: { top: 10, bottom: 16 },
      },
      datalabels: {
        color: '#fff',
        anchor: 'center',
        align: 'center',
        font: {
          weight: 'bold',
          size: 16,
        },
        formatter: (value) => value.toFixed(1),
      },
      tooltip: {
        enabled: true,
        callbacks: {
          label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(1)}`,
        },
      },
      annotation: validBettingLine !== null ? {
        annotations: {
          bettingLine: {
            type: 'line',
            yMin: validBettingLine,
            yMax: validBettingLine,
            borderColor: '#eab308',
            borderWidth: 3,
            borderDash: [6, 6],
            label: {
              display: true,
              content: 'Betting Line',
              color: '#eab308',
              font: {
                size: 14,
                weight: 'bold',
              },
              position: 'start',
              backgroundColor: 'rgba(0,0,0,0.7)',
              padding: 6,
            },
          },
        },
      } : {},
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(255,255,255,0.15)',
        },
        ticks: {
          color: '#fff',
          font: {
            size: 15,
            weight: 'bold',
          },
        },
      },
      x: {
        grid: {
          color: 'rgba(255,255,255,0.10)',
        },
        ticks: {
          color: '#fff',
          font: {
            size: 15,
            weight: 'bold',
          },
        },
      },
    },
    layout: {
      padding: { left: 0, right: 0, top: 0, bottom: 0 },
    },
    barPercentage: 0.55,
    categoryPercentage: 0.7,
  };

  const data = {
    labels: ['Season', 'Last 10 Games', `vs. ${opponentAbbr}`],
    datasets: [
      {
        label: category,
        data: [validSeasonAverage, validRecentAverage, validH2hAverage],
        backgroundColor: barColors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 8,
        barPercentage: 0.55,
        categoryPercentage: 0.7,
      },
    ],
  };

  return (
    <div className="chart-container" style={{ width: '100%', height: 320, boxSizing: 'border-box', margin: '0 auto' }}>
      <Bar options={options} data={data} plugins={[ChartDataLabels]} />
    </div>
  );
};

export default PlayerAveragesChart; 
