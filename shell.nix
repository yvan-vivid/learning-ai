{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.virtualenv
    pkgs.poethepoet
    pkgs.poetry
    pkgs.stdenv
  ];
  
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
