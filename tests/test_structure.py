from pathlib import Path
import pytest

@pytest.mark.structure
def test_example_readme_exists():
    readme_path = Path(__file__).parent / ".." / "example" / "README.md"
    assert readme_path.resolve().is_file(), "Missing example readme"

