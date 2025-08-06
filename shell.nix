{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = [
    pkgs.python313
    pkgs.python313Packages.virtualenv
    pkgs.poethepoet
    pkgs.poetry
    pkgs.stdenv
    pkgs.basedpyright
    pkgs.ruff
    pkgs.graphviz
    pkgs.uv
  ];

  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
