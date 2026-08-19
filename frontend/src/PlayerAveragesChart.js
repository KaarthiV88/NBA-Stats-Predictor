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

const MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";

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
      // The surrounding panel supplies the heading, so the chart itself stays
      // chrome-free and dense.
      title: { display: false },
      datalabels: {
        color: '#fff',
        anchor: 'center',
        align: 'center',
        font: {
          weight: '700',
          size: 13,
          family: MONO,
        },
        formatter: (value) => value.toFixed(1),
      },
      tooltip: {
        enabled: true,
        backgroundColor: '#2A2D35',
        borderColor: 'rgba(255,255,255,0.14)',
        borderWidth: 1,
        titleFont: { size: 12 },
        bodyFont: { size: 12, family: MONO },
        padding: 9,
        displayColors: false,
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
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
              display: true,
              content: `Line ${validBettingLine}`,
              color: '#eab308',
              font: {
                size: 11,
                weight: '700',
                family: MONO,
              },
              position: 'start',
              backgroundColor: 'rgba(24,26,32,0.9)',
              padding: 5,
            },
          },
        },
      } : {},
    },
    scales: {
      y: {
        beginAtZero: true,
        border: { display: false },
        grid: {
          color: 'rgba(255,255,255,0.06)',
        },
        ticks: {
          color: '#9ca3af',
          font: { size: 11, family: MONO },
          padding: 6,
        },
      },
      x: {
        border: { display: false },
        grid: { display: false },
        ticks: {
          color: '#9ca3af',
          font: { size: 11, weight: '600' },
          padding: 4,
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
        borderWidth: 1,
        borderRadius: 5,
        barPercentage: 0.55,
        categoryPercentage: 0.7,
      },
    ],
  };

  return (
    <div className="chart-container">
      <Bar options={options} data={data} plugins={[ChartDataLabels]} />
    </div>
  );
};

export default PlayerAveragesChart; 
