{
  description = "Flake for the Laser-AI development shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  outputs = { self, nixpkgs, ... }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      system = system;
      config.allowUnfree = true;
    };
  in {
    devShells.${system}.default = pkgs.mkShell {

      packages = [
        (pkgs.python313.withPackages(p: with p; [
          numpy
          chess
          torchWithCuda
          pandas
          requests
          fastapi
          fastapi-cli
        ]))
      ];

    };
  };
}
