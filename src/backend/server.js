const express = require('express');
const cors = require('cors');
const analysisRoutes = require('./routes/analysis');

const app = express();
const port = 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api', analysisRoutes);

// Basic error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});