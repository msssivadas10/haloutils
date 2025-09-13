from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import os, os.path, sys, subprocess, re, shutil

PACKAGE_NAME = "haloutils"
SRC_DIR      = "src"
BUILD_DIR    = "build_fortran"

def scan_fortran_sources(root_dir):
    # Return a dict mapping file paths to (module_name, used_modules).

    def parse_fortran_file(filepath):
        # Extract the module name and used modules from a Fortran 90 file.

        # Regex patterns (case-insensitive)
        module_re  = re.compile(r'^\s*module\s+([a-z_][a-z0-9_]*)', re.IGNORECASE)
        use_re     = re.compile(r'^\s*use\s+([a-z_][a-z0-9_]*)'   , re.IGNORECASE)
        modproc_re = re.compile(r'^\s*module\s+procedure'         , re.IGNORECASE)

        module_name  = None
        used_modules = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.split("!")[0].strip() # Strip inline comments
                if not line: continue
                if modproc_re.match(line): continue # Skip "module procedure"

                m = module_re.match(line) # module denfinition
                if m and module_name is None:
                    module_name = m.group(1).lower()
                    continue

                u = use_re.match(line) # use statement
                if u: used_modules.append(u.group(1).lower())

        return module_name, used_modules

    result = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith((".f90", ".F90")):
                fpath     = os.path.join(dirpath, fname)
                mod, uses = parse_fortran_file(fpath)
                if mod: result[fpath] = (mod, uses)
    return result

class BuildFortran90Shared(build_ext):

    def get_source_files(self):
        # Return a list of source files, sorted by dependency order. 

        # Build a dependency graph mapping file -> (module, [dependent files])
        mapping = scan_fortran_sources(SRC_DIR)
        module_to_file = {mod: path for path, (mod, _) in mapping.items()} # Reverse map: module -> file

        # Build resolved dependency graph
        graph = {}
        for path, (mod, uses) in mapping.items():
            resolved = []
            for u in uses:
                if u in module_to_file:
                    resolved.append(module_to_file[u])  # link to file
            graph[path] = (mod, resolved)

        # Perform topological sort on a dependency graph.
        visited = {}   # file -> "temporary" or "permanent"
        order   = []
        
        def dfs(node):
            if node in visited:
                if visited[node] == "temp":
                    raise ValueError(f"Cycle detected involving {node}")
                return
            visited[node] = "temp"
            _, deps = graph[node]
            for dep in deps:
                if dep.startswith("<unresolved:"): continue # ignore unresolved modules
                if dep not in graph: continue  # external file, ignore
                dfs(dep)
            visited[node] = "perm"
            order.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return order

    def run(self):

        if not os.path.exists(BUILD_DIR): os.mkdir(BUILD_DIR)

        # Detect platform-specific shared library extension
        if sys.platform.startswith("linux"): lib_extension = ".so"
        elif sys.platform == "darwin"      : lib_extension = ".dylib"
        elif sys.platform == "win32"       : lib_extension = ".dll"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
        
        # Shared library path
        if not os.path.exists(PACKAGE_NAME): os.mkdir(PACKAGE_NAME)
        lib_path = os.path.join(PACKAGE_NAME, "lib" + PACKAGE_NAME + lib_extension)

        # List source files
        files = self.get_source_files()

        # Compile files to shared library
        fc   = os.environ.get("FC", "gfortran")
        args = ( 
            os.environ.get("FCARGS", "").split() 
            or 
            [ "-shared", "-fPIC", "-J", BUILD_DIR, "-fopenmp", "-ffixed-line-length-0" ] 
        )
        print(f"using fortran compiler {fc} with args {args}"    , flush = True)
        print(f"compiling files {', '.join(files)} to {lib_path}", flush = True)
        subprocess.check_call([ fc, *args, *files, "-o", lib_path ])

        if os.path.exists(BUILD_DIR): shutil.rmtree(BUILD_DIR)

        return

setup(
    name     = PACKAGE_NAME,
    version  = "0.1",
    packages = [ PACKAGE_NAME ],
    cmdclass = { "build_ext": BuildFortran90Shared },
)