{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {

  packages = [
    (pkgs.python3.withPackages(p: with p; [
      numpy
      chess
      torch
      pandas
      requests
      fastapi
    ]))
  ];

}
