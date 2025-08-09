from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import git
import os
import ast
import glob

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Repo(BaseModel):
    url: str

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, file_path, file_to_id, repo_path):
        self.file_path = file_path
        self.file_to_id = file_to_id
        self.repo_path = repo_path
        self.edges = []
        self.imported_modules = {}

    def visit_Import(self, node):
        for alias in node.names:
            self.imported_modules[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.imported_modules[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.imported_modules:
                module_name = self.imported_modules[func_name]
                module_path = f"{module_name.replace('.', '/')}.py"
                if module_path in self.file_to_id:
                    source_id = self.file_to_id[os.path.relpath(self.file_path, self.repo_path)]
                    target_id = self.file_to_id[module_path]
                    self.edges.append({"source": source_id, "target": target_id})
        self.generic_visit(node)


@app.post("/api/analyze")
def analyze_repo(repo: Repo):
    repo_url = repo.url
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join("/tmp", repo_name)

    # Clone the repo
    if os.path.exists(repo_path):
        os.system(f"rm -rf {repo_path}")
    git.Repo.clone_from(repo_url, repo_path)

    # Analyze the repo
    nodes = []
    edges = []
    file_to_id = {}

    # Find all python files
    py_files = glob.glob(os.path.join(repo_path, "**/*.py"), recursive=True)

    for i, file_path in enumerate(py_files):
        relative_path = os.path.relpath(file_path, repo_path)
        nodes.append({"id": i, "name": relative_path})
        file_to_id[relative_path] = i

    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                tree = ast.parse(f.read())
                analyzer = CodeAnalyzer(file_path, file_to_id, repo_path)
                analyzer.visit(tree)
                edges.extend(analyzer.edges)
            except Exception as e:
                print(f"Could not parse {file_path}: {e}")

    # Remove duplicate edges
    unique_edges = [dict(t) for t in {tuple(d.items()) for d in edges}]

    return {"nodes": nodes, "edges": unique_edges}
