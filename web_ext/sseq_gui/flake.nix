{
  description = "sseq_gui flake devshell";

  inputs = {
    super.url = "path:../.."; # points to top-level flake
  };

  outputs = {super, ...}:
    super.flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import super.nixpkgs {inherit system;};
      fenixPkgs = super.fenix.packages.${system};

      rustToolchain = fenixPkgs.combine [
        super.defaultPackages.rustToolchain.${system}
        fenixPkgs.targets.wasm32-unknown-unknown.latest.toolchain
      ];

      pythonEnv = pkgs.python3.withPackages (ps: [
        ps.flake8
        ps.black
      ]);

      commonPackages = [
        rustToolchain
        super.defaultPackages.devTools.${system}

        pythonEnv
        pkgs.openssl
      ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
        # NixOS-specific packages for Playwright
        pkgs.playwright-test
        pkgs.playwright-driver
        pkgs.playwright-driver.browsers
      ];

      runTestScript = pkgs.writeShellScript "run-tests" ''
        set -euo pipefail

        echo "Configuring Playwright for NixOS..."
        export PLAYWRIGHT_BROWSERS_PATH=${pkgs.playwright-driver.browsers}
        export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true

        make lint
        make lint-playwright

        cargo install wasm-bindgen-cli --debug
        make lint-wasm
        make wasm

        make serve-wasm &
        (sleep 1 && NIXOS=1 make playwright)

        cargo build &&
        (target/debug/sseq_gui &
         (sleep 1 && NIXOS=1 make playwright))

        cargo build --features concurrent &&
        (target/debug/sseq_gui &
         (sleep 1 && NIXOS=1 make playwright))
      '';
    in {
      devShells.default = pkgs.mkShell {
        packages = commonPackages;

        shellHook = ''
          export RUST_LOG=info
          export PLAYWRIGHT_BROWSERS_PATH=${pkgs.playwright-driver.browsers}
          export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
        '';
      };

      apps.test = {
        type = "app";
        packages = commonPackages;
        program = toString runTestScript;
      };
    });
}
