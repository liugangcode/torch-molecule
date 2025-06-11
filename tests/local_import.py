import os
import ast

def files_importing_pandas(root_dir):
    pandas_files = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                source = open(path, encoding='utf8').read()
                tree = ast.parse(source, path)
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                # catch “import pandas” or “import pandas as pd”
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == 'pandas' or alias.name.startswith('pandas.'):
                            pandas_files.add(path)
                            break
                # catch “from pandas import …” or “from pandas.core import …”
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('pandas'):
                        pandas_files.add(path)
                        break
    return pandas_files

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # or replace with your path
    print(f"Project root: {project_root}")
    for f in sorted(files_importing_pandas(project_root)):
        print(f)
