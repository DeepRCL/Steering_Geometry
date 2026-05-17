from pathlib import Path

def get_project_dir() -> Path:
    current_dir = Path(__file__).parent.absolute()
    markers = ['.git', 'pyproject.toml']
    
    while current_dir != current_dir.parent:
        if any((current_dir / marker).exists() for marker in markers):
            return current_dir
        current_dir = current_dir.parent
    
    raise FileNotFoundError(
        "Could not find project root directory. "
        "Make sure you're running this from within the repe_ode project."
    )
    
__all__ = ['get_project_dir']