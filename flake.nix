{
  description = "Flake for the Laser-AI development shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: let
    pkgs = import nixpkgs {
      system = "x86_64-linux";
      #config.allowUnfree = true;
      #config.cudaSupport = true;
    };
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {

      packages = [
        (pkgs.python313.withPackages(p: with p; [
          numpy
          chess
          torch
          pandas
          requests
          fastapi
          fastapi-cli
        ]))
      ];

    };
  };
}

