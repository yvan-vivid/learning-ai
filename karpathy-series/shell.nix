{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "karpathy-ai-series-env";
  buildInputs = [
    pkgs.python310
    pkgs.python310Packages.virtualenv
  ];

  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
