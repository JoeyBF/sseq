{
  description = "ext-rs dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    fenix,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          (fenix.packages.${system}.complete.withComponents
            [
              "rustc"
              "cargo"
              "clippy"
              "rustfmt"
              "rust-src"
              "rust-analyzer"
              "miri"
            ])
          (pkgs.python3.withPackages (ps: [
            ps.black
            ps.pytest
          ]))
        ];

        shellHook = ''
          export RUST_LOG=info
        '';
      };
    });
}
