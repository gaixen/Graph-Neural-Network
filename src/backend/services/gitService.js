const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const tempDir = path.join(__dirname, '../db/temp_clones');

const cloneRepo = async (repoUrl) => {
    await fs.ensureDir(tempDir);
    const repoName = repoUrl.split('/').pop().replace('.git', '');
    const localPath = path.join(tempDir, repoName);

    // Clear out the directory if it already exists
    if (await fs.pathExists(localPath)) {
        await fs.remove(localPath);
    }
    
    const git = simpleGit();
    await git.clone(repoUrl, localPath);
    return localPath;
};

const getFileTree = async (localPath) => {
    const tree = [];
    const walk = async (dir, currentPath = '') => {
        const files = await fs.readdir(dir);
        for (const file of files) {
            const filePath = path.join(dir, file);
            const relativePath = path.join(currentPath, file);
            const stats = await fs.stat(filePath);

            // Ignore .git directory
            if (file === '.git') continue;

            if (stats.isDirectory()) {
                await walk(filePath, relativePath);
            } else {
                tree.push({
                    path: relativePath.replace(/\\/g, '/'), // Ensure consistent path separators
                    type: 'blob', // Match GitHub API terminology
                });
            }
        }
    };
    await walk(localPath);
    return tree;
};

const getFileContent = async(localPath, filePath) => {
    const fullPath = path.join(localPath, filePath);
    if (!await fs.pathExists(fullPath)) {
        throw new Error('File does not exist in the cloned repository.');
    }
    return fs.readFile(fullPath, 'utf-8');
}


module.exports = {
    cloneRepo,
    getFileTree,
    getFileContent,
};
