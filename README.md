# CodeGraph: Visualizing Code Repositories

This project is a full-stack web application that takes a GitHub repository URL, analyzes the codebase, and generates an interactive graph visualization of the code's structure. This tool helps developers understand file dependencies, function calls, and the overall architecture of a project.

## Overview

The core of this project is to provide a visual representation of a codebase to make it easier to understand and navigate. By parsing the source code and representing it as a graph, developers can quickly see how different parts of the application are connected, identify potential issues, and get a high-level overview of the project's architecture.

## Features

-   **GitHub Repository Input:** A simple UI to enter the URL of a public GitHub repository.
-   **Code Analysis Engine (Backend):**
    -   Clones the repository.
    -   Parses the source code to identify files and their dependencies.
    -   Identifies frontend and backend code.
    -   Traces API calls from the frontend to backend endpoints.
-   **Interactive Graph Visualization:**
    -   Displays the code as a graph with files as nodes and dependencies as edges.
    -   Uses different colors to distinguish between frontend and backend files.
    -   Highlights API calls to show frontend-backend interaction.

## Architecture

The application is built with a modern full-stack architecture, consisting of a React frontend and a FastAPI backend.

```
+-----------------+      +----------------------+      +-----------------+
|                 |      |                      |      |                 |
|  React Frontend |----->|    FastAPI Backend   |----->| Cloned Git Repo |
| (Visualization) |      | (API & Code Analysis)|      |                 |
|                 |      |                      |      |                 |
+-----------------+      +----------------------+      +-----------------+
```

-   **Frontend:** A responsive user interface built with **React**. It provides an input for the GitHub URL and uses a graph visualization library to render the codebase graph.
-   **Backend:** A high-performance API built with **FastAPI** (Python). It handles cloning the repository, parsing the code, and serving the graph data to the frontend.

## Technology Stack

-   **Frontend:** React, JavaScript
-   **Backend:** FastAPI, Python, GitPython
-   **Visualization:** A suitable graph visualization library for React.