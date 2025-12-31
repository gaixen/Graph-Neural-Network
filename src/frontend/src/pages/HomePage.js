import React, { useState, useMemo, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import ReactMarkdown from 'react-markdown';
import './HomePage.css';

function HomePage() {
  const [repoUrl, setRepoUrl] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [explanation, setExplanation] = useState('');
  const [isExplanationLoading, setIsExplanationLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);

  const graphData = useMemo(() => {
    if (!analysisResult || !analysisResult.fileTree) return { nodes: [], links: [] };

    const nodes = new Map();
    const links = new Set();

    const addNode = (id, name, type, item) => {
      if (!nodes.has(id)) {
        nodes.set(id, { id, name, type, item });
      }
    };

    addNode('root', 'Repo Root', 'directory');

    analysisResult.fileTree.forEach(item => {
      const pathParts = item.path.split('/');
      let currentPath = 'root';

      pathParts.forEach((part, i) => {
        const isLastPart = i === pathParts.length - 1;
        const childPath = isLastPart ? item.path : pathParts.slice(0, i + 1).join('/');
        
        addNode(childPath, part, isLastPart ? item.type : 'directory', item);
        
        if (currentPath) {
            links.add(JSON.stringify({ source: currentPath, target: childPath }));
        }
        
        currentPath = childPath;
      });
    });

    return { 
        nodes: Array.from(nodes.values()), 
        links: Array.from(links).map(link => JSON.parse(link)) 
    };
  }, [analysisResult]);

  const handleNodeClick = useCallback((node) => {
    if (node.type === 'blob') { // 'blob' is for files
      setSelectedNode(node);
      setIsExplanationLoading(true);
      setExplanation('');

      // Find the explanation from the initial analysis data
      const fileAnalysis = analysisResult?.analysis?.find(a => a.path === node.id);

      if (fileAnalysis) {
        setExplanation(fileAnalysis.explanation);
        setIsExplanationLoading(false);
      } else {
        setExplanation('No explanation found for this file.');
        setIsExplanationLoading(false);
      }
    }
  }, [analysisResult]); // Added analysisResult to the dependency array

  const handleAnalyze = () => {
    setLoading(true);
    setError(null);
    setAnalysisResult(null);
    setExplanation('');
    setSelectedNode(null);
    fetch('http://localhost:3001/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoUrl }),
    })
      .then(res => {
        if (!res.ok) return res.json().then(err => { throw new Error(err.error || `HTTP error! status: ${res.status}`) });
        return res.json();
      })
      .then(data => setAnalysisResult(data))
      .catch(error => setError(error.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="homepage-container">
      <div className="main-content">
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
          {loading && <div className="status-overlay">Analyzing Repository...</div>}
          {error && <div className="status-overlay error">Error: {error}</div>}
          {analysisResult && (
            <ForceGraph2D
              graphData={graphData}
              nodeLabel="name"
              nodeAutoColorBy="type"
              linkDirectionalParticles={1}
              linkDirectionalParticleWidth={1.5}
              onNodeClick={handleNodeClick}
              nodeCanvasObject={(node, ctx, globalScale) => {
                const label = node.name;
                const fontSize = 12 / globalScale;
                ctx.font = `${fontSize}px Sans-Serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = node === selectedNode ? 'red' : '#333';
                ctx.fillText(label, node.x, node.y + 8);
              }}
            />
          )}
        </div>
      </div>
      <aside className="sidebar">
        <h2>AI Code Explanation</h2>
        <div className="explanation-content">
          {isExplanationLoading && <p>Loading explanation...</p>}
          {explanation ? (
            <ReactMarkdown>{explanation}</ReactMarkdown>
          ) : (
            !isExplanationLoading && <p>Click on a file node in the graph to see an AI-powered explanation.</p>
          )}
        </div>
      </aside>
    </div>
  );
}

export default HomePage;
