import os
import tempfile
import git
from pathlib import Path
import mimetypes
from typing import Tuple, List

def github_to_markdown(repo_url: str) -> str:
    """
    Clone a GitHub repository and convert its contents to a markdown format with
    separate sections for structure and content.
    
    Args:
        repo_url (str): HTTPS URL to the GitHub repository
        
    Returns:
        str: Path to the generated markdown file
    """
    def is_text_file(file_path: Path) -> bool:
        """Check if a file is a text file based on its mimetype and extension."""
        if file_path.suffix == '.html':  # Ignore HTML files
            return False
            
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            return file_path.suffix in {'.md', '.txt', '.py', '.js', '.java', '.cpp', 
                                      '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift',
                                      '.kt', '.kts', '.sh', '.bash', '.yaml', '.yml',
                                      '.json', '.xml', '.css', '.scss', '.sql',
                                      '.conf', '.cfg', '.ini', '.toml', '.env'}
        return mime_type.startswith('text/')

    def should_ignore(path: Path) -> bool:
        """Check if a path should be ignored."""
        ignore_patterns = {'.git', '.github', '__pycache__', '.pytest_cache',
                         '.idea', '.vscode', 'node_modules', '.DS_Store'}
        return any(pattern in path.parts for pattern in ignore_patterns)

    def process_directory(directory: Path) -> Tuple[List[str], List[Tuple[str, Path]]]:
        """
        Recursively process a directory and return its structure and file paths.
        
        Args:
            directory (Path): Directory path to process
            
        Returns:
            Tuple[List[str], List[Tuple[str, Path]]]: 
                - List of strings representing directory structure
                - List of tuples containing (relative path, absolute path) for files
        """
        structure = []
        files_to_process = []
        base_path = Path(directory)

        def _recurse(current_path: Path, level: int = 0):
            indent = "  " * level
            paths = sorted(current_path.iterdir(), 
                         key=lambda x: (not x.is_dir(), x.name.lower()))

            for path in paths:
                if should_ignore(path):
                    continue

                rel_path = str(path.relative_to(base_path))

                if path.is_dir():
                    structure.append(f"{indent}- 📁 **{path.name}/**")
                    _recurse(path, level + 1)
                else:
                    if is_text_file(path):
                        structure.append(f"{indent}- 📄 {path.name}")
                        files_to_process.append((rel_path, path))

        _recurse(directory)
        return structure, files_to_process

    def generate_content(structure: List[str], files: List[Tuple[str, Path]], repo_name: str) -> str:
        """Generate the final markdown content with separate sections."""
        content = [
            f"# {repo_name}\n",
            "## Repository Structure\n",
            "```",
            *structure,
            "```\n",
            "## File Contents\n"
        ]

        for rel_path, abs_path in files:
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    if file_content.strip():  # Only include non-empty files
                        content.extend([
                            f"### 📄 `{rel_path}`\n",
                            f"```{abs_path.suffix[1:] if abs_path.suffix else ''}",
                            file_content.rstrip(),
                            "```\n"
                        ])
            except UnicodeDecodeError:
                content.extend([
                    f"### 📄 `{rel_path}`\n",
                    "*[Binary file or encoding error]*\n"
                ])

        return '\n'.join(content)

    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone repository
            repo = git.Repo.clone_from(repo_url, temp_dir, branch='main')
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            
            # Process repository
            structure, files = process_directory(Path(temp_dir))
            markdown_content = generate_content(structure, files, repo_name)
            
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
    # Example repository URL
    #repo_url = "https://github.com/phunterlau/paper_without_code.git"
    repo_url = sys.argv[1]
    if not repo_url.endswith('.git'):
        repo_url += '.git'
    output_file = github_to_markdown(repo_url)
    print(f"Markdown file generated: {output_file}")