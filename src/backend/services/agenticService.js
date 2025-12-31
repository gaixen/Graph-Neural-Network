require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize the Gemini client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

const explainCode = async (filePath, fileContent) => {
    const prompt = `
        Analyze the following code file and provide a detailed explanation in Markdown format.

        **File Path:** ${filePath}

        **Code:**
        \`\`\`
        ${fileContent}
        \`\`\`

        **Analysis should include:**
        1.  **Summary:** A high-level overview of the file's purpose.
        2.  **Core Functionality:** A description of the key functions, classes, and logic.
        3.  **Dependencies:** Any imported modules or external dependencies.
        4.  **Potential Interactions:** How this file might interact with other parts of the application.
    `;

    try {
        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = await response.text();
        return text;
    } catch (error) {
        console.error('Error calling Gemini API:', error);
        return 'Error: Could not get explanation from AI service.';
    }
};

module.exports = {
    explainCode,
};