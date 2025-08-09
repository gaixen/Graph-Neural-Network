const axios = require('axios');

// Recursive function to fetch the entire file tree
const getTree = async (owner, repo, tree_sha) => {
    try {
        const response = await axios.get(`https://api.github.com/repos/${owner}/${repo}/git/trees/${tree_sha}?recursive=1`);
        return response.data.tree;
    } catch (error) {
        console.error('Error fetching tree:', error.response ? error.response.data : error.message);
        throw new Error('Could not fetch repository tree.');
    }
};

// Function to get the default branch
const getDefaultBranch = async (owner, repo) => {
    try {
        const response = await axios.get(`https://api.github.com/repos/${owner}/${repo}`);
        return response.data.default_branch;
    } catch (error) {
        console.error('Error fetching repo details:', error.response ? error.response.data : error.message);
        throw new Error('Could not fetch repository details.');
    }
};

const analyzeRepo = async (repoUrl) => {
    // Simple regex to extract owner/repo from various GitHub URL formats
    const match = repoUrl.match(/github\.com\/([^\/]+\/[^\/]+)(\.git)?/);
    if (!match) {
        throw new Error('Invalid GitHub repository URL');
    }

    const [owner, repo] = match[1].split('/');

    // 1. Get the default branch name
    const branch = await getDefaultBranch(owner, repo);

    // 2. Get the commit SHA for the latest commit on that branch
    const commitResponse = await axios.get(`https://api.github.com/repos/${owner}/${repo}/branches/${branch}`);
    const tree_sha = commitResponse.data.commit.sha;
    
    // 3. Get the file tree using the commit's SHA
    const tree = await getTree(owner, repo, tree_sha);

    return tree;
};

module.exports = {
    analyzeRepo,
};
