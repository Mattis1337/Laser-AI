{
  description = "Flake for the Laser-AI development shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      system = system;
      config.allowUnfree = true;
    };
  in {
    devShells.${system} = {
      default = pkgs.mkShell {

        packages = [
          (pkgs.python312.withPackages(p: with p; [
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
      cuda = pkgs.mkShell {

        packages = [
          (pkgs.python312.withPackages(p: with p; [
            numpy
            chess
            torch-bin  # precompiled PyTorch with CUDA support
            pandas
            requests
            fastapi
            fastapi-cli
          ]))
        ];

      };
    };
  };
}
