{
  description = "LLaMa2 in Haskell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              haskell = prev.haskell // {
                packages = prev.haskell.packages // {
                  ghc948 = prev.haskell.packages.ghc948.override {
                    overrides = self: super: {
                      # You can add package overrides here if needed
                    };
                  };
                };
              };
            })
          ];
        };

        haskellPackages = pkgs.haskell.packages.ghc948;

        packageName = "llama2";
      in {
        packages.${packageName} = haskellPackages.callCabal2nix packageName ./. {};

        defaultPackage = self.packages.${system}.${packageName};

        devShell = pkgs.mkShell {
          buildInputs = with haskellPackages; [
            ghc
            cabal-install
            haskell-language-server
            # Add any other development tools you need
          ];
        };
      }
    );
}