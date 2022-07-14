import sphinx.cmd.build as build

build.build_main(argv=["-b", "html", "docs", "_build"])