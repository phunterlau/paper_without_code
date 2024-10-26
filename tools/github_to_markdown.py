import os
import tempfile
import git
from pathlib import Path
import mimetypes
from typing import Tuple, List
import re
import json
from urllib.parse import urlparse

def github_to_markdown(repo_url: str) -> str:
    """
    Clone a GitHub repository and convert its contents to a markdown format with
    separate sections for structure and content.
    
    Args:
        repo_url (str): HTTPS URL to the GitHub repository or subdirectory
        
    Returns:
        str: Path to the generated markdown file
    """
    def convert_notebook_to_python(notebook_path: Path) -> str:
        """
        Convert Jupyter notebook to Python code with markdown cells as comments.
        
        Args:
            notebook_path (Path): Path to the .ipynb file
            
        Returns:
            str: Converted Python code
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            python_code = []
            cell_counter = 0

            for cell in notebook['cells']:
                cell_counter += 1
                
                # Handle markdown cells
                if cell['cell_type'] == 'markdown':
                    markdown_content = ''.join(cell['source'])
                    # Convert markdown content to comments
                    python_code.extend([
                        f"\n#{'=' * 78}",
                        f"# Markdown Cell {cell_counter}:",
                        f"#{'=' * 78}"
                    ])
                    for line in markdown_content.split('\n'):
                        # Preserve empty lines in markdown
                        if line.strip():
                            python_code.append(f"# {line}")
                        else:
                            python_code.append("#")
                
                # Handle code cells
                elif cell['cell_type'] == 'code':
                    code_content = ''.join(cell['source'])
                    if code_content.strip():  # Skip empty code cells
                        python_code.extend([
                            f"\n#{'=' * 78}",
                            f"# Code Cell {cell_counter}:",
                            f"#{'=' * 78}",
                            code_content.rstrip()
                        ])

            return '\n'.join(python_code)

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            return f"# Error converting notebook: {str(e)}"

    def parse_github_url(url: str) -> Tuple[str, str, str]:
        """Parse GitHub URL to extract repo URL and subdirectory path."""
        pattern = r'https://github\.com/([^/]+/[^/]+)(?:/(?:tree|blob)/[^/]+)?(/.*)?'
        match = re.match(pattern, url)
        
        if not match:
            raise ValueError("Invalid GitHub URL")
            
        repo_path = match.group(1)
        subdir = (match.group(2) or '').strip('/')
        repo_name = repo_path.split('/')[-1]
        
        repo_url = f'https://github.com/{repo_path}'
        if not repo_url.endswith('.git'):
            repo_url += '.git'
            
        return repo_url, subdir, repo_name

    def count_file_lines(file_path: Path) -> int:
        """Count number of lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except (UnicodeDecodeError, OSError):
            return 0

    def is_excluded_file(file_path: Path) -> bool:
        """Check if a file should be excluded from content processing."""
        excluded_extensions = {'.html', '.txt'}  # Removed .ipynb from exclusion
        return file_path.suffix in excluded_extensions

    def is_text_file(file_path: Path) -> bool:
        """Check if a file is a text file based on its mimetype and extension."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            return file_path.suffix in {'.md', '.txt', '.py', '.js', '.java', '.cpp', 
                                      '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift',
                                      '.kt', '.kts', '.sh', '.bash', '.yaml', '.yml',
                                      '.json', '.xml', '.css', '.scss', '.sql',
                                      '.conf', '.cfg', '.ini', '.toml', '.env',
                                      '.html', '.ipynb'}
        return mime_type.startswith('text/')

    def should_ignore(path: Path) -> bool:
        """Check if a path should be ignored completely from tree."""
        ignore_patterns = {'.git', '.github', '__pycache__', '.pytest_cache',
                         '.idea', '.vscode', 'node_modules', '.DS_Store'}
        return any(pattern in path.parts for pattern in ignore_patterns)

    def process_directory(directory: Path, target_subdir: str = '') -> Tuple[List[str], List[Tuple[str, Path]], set]:
        """Recursively process a directory and return its structure and file paths."""
        structure = []
        files_to_process = []
        included_paths = set()
        base_path = Path(directory)
        target_path = base_path / target_subdir if target_subdir else base_path

        def _recurse(current_path: Path, level: int = 0):
            indent = "  " * level
            paths = sorted(current_path.iterdir(), 
                         key=lambda x: (not x.is_dir(), x.name.lower()))

            for path in paths:
                if should_ignore(path):
                    continue

                rel_path = str(path.relative_to(base_path))
                is_in_target = target_subdir == '' or rel_path.startswith(target_subdir)

                if path.is_dir():
                    line_info = ""
                    prefix = "📂" if is_in_target else "📁"
                    structure.append(f"{indent}- {prefix} **{path.name}/**{line_info}")
                    if is_in_target:
                        included_paths.add(rel_path)
                    _recurse(path, level + 1)
                else:
                    if is_text_file(path):
                        line_count = count_file_lines(path)
                        line_info = f" `({line_count} lines)`"
                        
                        if is_in_target and not is_excluded_file(path):
                            prefix = "📝"
                            included_paths.add(rel_path)
                            files_to_process.append((rel_path, path))
                        else:
                            prefix = "📄"
                            
                        structure.append(f"{indent}- {prefix} {path.name}{line_info}")
                    else:
                        prefix = "📄"
                        structure.append(f"{indent}- {prefix} {path.name}")

        _recurse(base_path)
        return structure, files_to_process, included_paths

    def generate_content(structure: List[str], files: List[Tuple[str, Path]], repo_name: str, included_paths: set) -> str:
        """Generate the final markdown content with separate sections."""
        content = [
            f"# {repo_name}\n",
            "## Repository Structure\n",
            "Legend:\n" +
            "- 📂/📝 - Included content in markdown\n" +
            "- 📁/📄 - Excluded or non-text content\n" +
            "```",
            *structure,
            "```\n",
            "## File Contents\n"
        ]

        for rel_path, abs_path in files:
            try:
                if abs_path.suffix == '.ipynb':
                    # Convert notebook to Python
                    file_content = convert_notebook_to_python(abs_path)
                    content.extend([
                        f"### 📝 `{rel_path}` (converted from notebook)\n",
                        "```python",
                        file_content.rstrip(),
                        "```\n"
                    ])
                else:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        if file_content.strip():  # Only include non-empty files
                            content.extend([
                                f"### 📝 `{rel_path}`\n",
                                f"```{abs_path.suffix[1:] if abs_path.suffix else ''}",
                                file_content.rstrip(),
                                "```\n"
                            ])
            except UnicodeDecodeError:
                content.extend([
                    f"### 📝 `{rel_path}`\n",
                    "*[Binary file or encoding error]*\n"
                ])

        return '\n'.join(content)

    # Parse the input URL
    repo_url, subdir, repo_name = parse_github_url(repo_url)

    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone repository
            repo = git.Repo.clone_from(repo_url, temp_dir, branch='main')
            
            # Process repository
            structure, files, included_paths = process_directory(Path(temp_dir), subdir)
            markdown_content = generate_content(structure, files, repo_name, included_paths)
            
            # Create output file
            output_path = f"{repo_name}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return output_path

        except git.exc.GitCommandError as e:
            return f"Error cloning repository: {str(e)}"
        except Exception as e:
            return f"Error processing repository: {str(e)}"

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <github_url>")
        sys.exit(1)
        
    repo_url = sys.argv[1]
    output_file = github_to_markdown(repo_url)
    print(f"Markdown file generated: {output_file}")