project = "DeepGNN"
copyright = "2022, Microsoft"
author = "DeepGNN team"

# The short X.Y version
version = "0.1"
# The full version, including alpha/beta/rc tags
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
]

templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"
language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = None
html_theme = "alabaster"
html_static_path = ["../_build/_static"]
htmlhelp_basename = "DeepGNNdoc"
man_pages = [(master_doc, "deepgnn", "DeepGNN Documentation", [author], 1)]

todo_include_todos = False
