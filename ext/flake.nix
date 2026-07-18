{
  description = "ext-rs dev shell";

  inputs = {
    super.url = "path:.."; # points to top-level flake
  };

  outputs = {super, ...}:
    super.flake-utils.lib.eachDefaultSystem (system: let
      # Allow CUDA (unfree in nixpkgs). Scoped to the CUDA / NVIDIA prefix so
      # we don't accidentally unfree-allow anything else. nixpkgs splits the
      # toolkit into many sub-derivations (cuda_nvcc, cuda_cudart, cuda-merged,
      # cuda_cuobjdump, libcublas, ...) — listing them individually is whack-
      # a-mole, so we match by prefix.
      pkgs = import super.nixpkgs {
        inherit system;
        config.allowUnfreePredicate = pkg:
          let
            lib = super.nixpkgs.lib;
            name = lib.getName pkg;
          in
            lib.hasPrefix "cuda" name
            || lib.hasPrefix "libcu" name
            || lib.hasPrefix "libnv" name
            || lib.hasPrefix "libnpp" name;
      };

      # CUDA is unfree, so it needs its own nixpkgs instance. Only the `gpu` dev
      # shell pulls it in — the default shell and `nix run .#test` stay CUDA-free.
      cudaPkgs = import super.nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

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

      # CUDA toolkit serving both GPU backends. algebra's CubeCL `cuda` backend
      # JIT-compiles kernels with NVRTC — it needs `CUDA_PATH` pointing at a tree
      # with `include/` (NVRTC `--include-path`) plus libnvrtc. fp-cuda's Hopper
      # wgmma.b1 kernel needs nvcc + headers at build time. The monolithic
      # `cudatoolkit` gives one prefix with both headers and libs. Kept out of
      # `commonPackages` (and the default shell) so contributors and the
      # `apps.test`/CI closure don't fetch the multi-GB unfree CUDA tree for the
      # opt-in backends. cudarc dlopens libcuda at runtime (driver at
      # /run/opengl-driver/lib, not nixpkgs), so only running — not building the
      # Rust — needs the host driver.
      cudatoolkit = cudaPkgs.cudaPackages.cudatoolkit;
    in {
      devShells.default = pkgs.mkShell {
        packages = commonPackages;
        shellHook = ''
          export RUST_LOG=info
        '';
      };

      # GPU dev shell: `nix develop .#gpu`. Adds the CUDA toolkit (nvcc + headers)
      # and points the loader at both it and the host driver's libcuda.
      devShells.gpu = pkgs.mkShell {
        packages = commonPackages ++ [cudatoolkit];
        shellHook = ''
          export RUST_LOG=info
          export CUDA_PATH="${cudatoolkit}"
          export LD_LIBRARY_PATH="${cudatoolkit}/lib:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
