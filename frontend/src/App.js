import React, { useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import './App.css';

function App() {
  const [data, setData] = useState({ nodes: [], links: [] });
  const [repoUrl, setRepoUrl] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAnalyze = () => {
    setLoading(true);
    fetch('http://localhost:8000/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: repoUrl }),
    })
      .then(res => res.json())
      .then(data => {
        const links = data.edges.map(edge => ({
          source: edge.source,
          target: edge.target,
        }));
        setData({ nodes: data.nodes, links });
        setLoading(false);
      })
      .catch(error => {
        console.error('Error:', error);
        setLoading(false);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>CodeGraph: Visualizing Code Repositories</h1>
      </header>
      <div className="controls">
        <input
          type="text"
          value={repoUrl}
          onChange={e => setRepoUrl(e.target.value)}
          placeholder="Enter GitHub repository URL"
        />
        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
      <div className="graph-container">
        <ForceGraph2D
          graphData={data}
          nodeLabel="name"
          nodeAutoColorBy="name"
          linkDirectionalParticles={1}
        />
      </div>
    </div>
  );
}

export default App;
