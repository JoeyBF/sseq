{
  description = "ext-rs dev shell";

  inputs = {
    super.url = "path:.."; # points to top-level flake
  };

  outputs = {super, ...}:
    super.flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import super.nixpkgs {inherit system;};

      pythonEnv = pkgs.python3.withPackages (ps: [
        ps.black
        ps.pytest
      ]);

      commonPackages =
        [
          super.defaultPackages.rustToolchain.${system}

          pythonEnv

          pkgs.cargo-cache
          pkgs.cargo-criterion
          pkgs.cargo-flamegraph
          pkgs.cargo-nextest
          pkgs.perf
        ]
        ++ super.defaultPackages.devTools.${system};

      # Runtime libraries that `winit` / `glutin` dlopen for the optional `3d` visualizer
      # (`cargo run -p sseq --features 3d --example viz3d`, and the offscreen screenshot path).
      # Nix shells don't expose the host's system libraries, so these must be on
      # `LD_LIBRARY_PATH` or creating the window / GL context fails to initialize. `wayland`
      # covers Wayland compositors such as Hyprland; the `xorg.*` set covers X11 / XWayland.
      guiLibs = with pkgs; [
        libGL
        libxkbcommon
        wayland
        xorg.libX11
        xorg.libXcursor
        xorg.libXi
        xorg.libXrandr
      ];
    in {
      devShells.default = pkgs.mkShell {
        packages = commonPackages;
        shellHook = ''
          export RUST_LOG=info
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath guiLibs}:''${LD_LIBRARY_PATH:-}"
        '';
      };

      apps.test = {
        type = "app";
        packages = commonPackages;
        program = toString (pkgs.writeShellScript "run-tests" ''
          set -euo pipefail

          export RUSTFLAGS="-D warnings"
          export RUSTDOCFLAGS="-D warnings"

          just lint
          just test
          just benchmarks
          just benchmarks-nassau
          just benchmarks-concurrent
          just miri
        '');
      };
    });
}
