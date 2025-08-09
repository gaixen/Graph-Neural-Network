const githubService = require('../services/githubService');

const analyzeRepository = async (req, res) => {
    const { repoUrl } = req.body;
    if (!repoUrl) {
        return res.status(400).json({ error: 'repoUrl is required' });
    }

    try {
        const tree = await githubService.analyzeRepo(repoUrl);
        res.json(tree);
    } catch (error) {
        // Distinguish between user error (bad URL) and server error
        if (error.message === 'Invalid GitHub repository URL') {
            return res.status(400).json({ error: error.message });
        }
        res.status(500).json({ error: error.message });
    }
};

module.exports = {
    analyzeRepository,
};
