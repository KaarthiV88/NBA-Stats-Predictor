// Single source of truth for the backend origin.
//
// Local development falls back to the Flask dev server on :5001. In a deployed
// build, set REACT_APP_API_URL at build time to point at wherever the Python
// backend is hosted (Create React App inlines REACT_APP_* vars during `npm run
// build`, so this is baked into the bundle -- it is not read at runtime).
export const API_BASE = (process.env.REACT_APP_API_URL || 'http://localhost:5001').replace(/\/$/, '');

export const apiUrl = (path) => `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
