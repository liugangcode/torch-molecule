import os
import ast

# List of packages from dependencies in pyproject.toml
packages = [
    'click',
    'joblib',
    'networkx',
    'yaml',  # PyYAML
    'sklearn',
    'scipy',
]

def files_importing_package(root_dir, package):
    package_files = set()
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
                # catch "import package" or "import package as alias"
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == package or alias.name.startswith(package + '.'):
                            package_files.add(path)
                            break
                # catch "from package import …" or "from package.module import …"
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith(package):
                        package_files.add(path)
                        break
    return package_files

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # or replace with your path
    print(f"Project root: {project_root}")
    
    for package in packages:
        print(f"\n=== Files importing {package} ===")
        files = files_importing_package(project_root, package)
        if files:
            for f in sorted(files):
                print(f)
        else:
            print(f"No files found importing {package}")
