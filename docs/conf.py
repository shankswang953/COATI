project = "COATI"
author = "COATI contributors"
copyright = "2026, COATI contributors"
release = "0.1.0"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "furo"
html_title = "COATI documentation"
html_logo = "_static/coati-logo.png"
html_static_path = ["_static"]
html_css_files = ["coati.css"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme_options = {
    "light_css_variables": {"color-brand-primary": "#006bc4", "color-brand-content": "#006bc4"},
    "dark_css_variables": {"color-brand-primary": "#46baff", "color-brand-content": "#46baff"},
    "source_repository": "https://github.com/shankswang953/COATI/",
    "source_branch": "main", "source_directory": "docs/",
}
