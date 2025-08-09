import React, { useState, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

function HomePage() {
  const [repoUrl, setRepoUrl] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const graphData = useMemo(() => {
    if (!analysisResult) return { nodes: [], links: [] };

    const nodes = new Map();
    const links = new Set();

    // Helper to add a node if it doesn't exist
    const addNode = (id, name, type) => {
      if (!nodes.has(id)) {
        nodes.set(id, { id, name, type });
      }
    };

    // Add root node
    addNode('root', 'Repository Root', 'directory');

    analysisResult.forEach(item => {
      const pathParts = item.path.split('/');
      let currentPath = 'root';

      pathParts.forEach((part, i) => {
        const isLastPart = i === pathParts.length - 1;
        const childPath = isLastPart ? item.path : pathParts.slice(0, i + 1).join('/');
        
        addNode(childPath, part, isLastPart ? item.type : 'directory');
        
        links.add({ source: currentPath, target: childPath });
        currentPath = childPath;
      });
    });

    return { nodes: Array.from(nodes.values()), links: Array.from(links) };
  }, [analysisResult]);

  const handleAnalyze = () => {
    setLoading(true);
    setError(null);
    setAnalysisResult(null);

    fetch('http://localhost:3001/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ repoUrl }),
    })
      .then(res => {
        if (!res.ok) {
          return res.json().then(err => { throw new Error(err.error || `HTTP error! status: ${res.status}`) });
        }
        return res.json();
      })
      .then(data => {
        setAnalysisResult(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error:', error);
        setError(error.message);
        setLoading(false);
      });
  };

  return (
    <>
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
        {loading && <div>Analyzing...</div>}
        {error && <div className="error">Error: {error}</div>}
        {analysisResult && (
          <ForceGraph2D
            graphData={graphData}
            nodeLabel="name"
            nodeAutoColorBy="type"
            linkDirectionalParticles={1}
            linkDirectionalParticleWidth={1.5}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const label = node.name;
              const fontSize = 12 / globalScale;
              ctx.font = `${fontSize}px Sans-Serif`;
              const textWidth = ctx.measureText(label).width;
              const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); // some padding

              ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
              ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);

              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = '#333';
              ctx.fillText(label, node.x, node.y);
            }}
          />
        )}
      </div>
    </>
  );
}

export default HomePage;
