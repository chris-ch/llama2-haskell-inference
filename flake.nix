{
  description = "LLaMa2 in Haskell";

  inputs = {
    nixpkgs.url = "github:NixOS//nixpkgs/24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        haskellPackages = pkgs.haskell.packages.ghc948;

        packageName = "llama2";
      in {
        packages.${packageName} = haskellPackages.callCabal2nix packageName ./. {};

        defaultPackage = self.packages.${system}.${packageName};

        devShell = pkgs.mkShell {
          buildInputs =  with pkgs; [
            haskellPackages.ghc
            haskellPackages.cabal-install
            haskellPackages.haskell-language-server
            zlib.dev
            # Add other packages you need in your development environment
          ];
          shellHook = ''
            echo "GHC version: $(ghc --version)"
            echo "Cabal version: $(cabal --version)"
          '';
        };
      }
    );
}