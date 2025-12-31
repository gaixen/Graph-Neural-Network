const gitService = require('../services/gitService');
const agenticService = require('../services/agenticService');

const analyzeRepository = async (req, res) => {
    const { repoUrl } = req.body;
    if (!repoUrl) {
        return res.status(400).json({ error: 'repoUrl is required' });
    }

    try {
        // 1. Clone the repository
        const localPath = await gitService.cloneRepo(repoUrl);

        // 2. Get the file tree
        const fileTree = await gitService.getFileTree(localPath);

        // 3. Analyze each file in the tree
        const analysisResults = [];
        for (const file of fileTree) {
            if (file.type === 'blob') { // Ensure we only process files
                const fileContent = await gitService.getFileContent(localPath, file.path);
                const explanation = await agenticService.explainCode(file.path, fileContent);
                analysisResults.push({
                    path: file.path,
                    explanation: explanation,
                });
            }
        }

        res.json({
            fileTree: fileTree,
            analysis: analysisResults,
        });

    } catch (error) {
        console.error('Error during repository analysis:', error);
        res.status(500).json({ error: 'Failed to analyze repository.' });
    }
};

module.exports = {
    analyzeRepository,
};
