{
  description = "devel-tools — Git-aware ctags indexing, live watchers, and LanceDB semantic search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        inherit (pkgs) lib;
        python = pkgs.python312;

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
        lockedOverlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Disable auto-patchelf for torch and nvidia wheels so their $ORIGIN RPATHs are preserved
        pyprojectOverrides = final: prev:
          lib.genAttrs
            (builtins.filter (n: lib.hasPrefix "nvidia-" n) (builtins.attrNames prev) ++ [ "torch" ])
            (n: prev.${n}.overrideAttrs (_: { dontAutoPatchelf = true; }));

        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          lockedOverlay
          pyprojectOverrides
        ]);

        runtimeEnv = pythonSet.mkVirtualEnv "devel-tools-env" workspace.deps.default;
        runtimeLibs = "${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:/run/opengl-driver/lib";

        # Runtime wrapper with universal-ctags, git, and offline HF environment
        develTools = pkgs.runCommand "devel-tools"
          { nativeBuildInputs = [ pkgs.makeWrapper ]; }
          ''
            mkdir -p $out/bin
            for b in devel-nav devel-watch devel-sync devel-search devel-tags devel-status; do
              if [ -f "${runtimeEnv}/bin/$b" ]; then
                makeWrapper "${runtimeEnv}/bin/$b" "$out/bin/$b" \
                  --prefix PATH : "${lib.makeBinPath [ pkgs.universal-ctags pkgs.git pkgs.coreutils ]}" \
                  --prefix LD_LIBRARY_PATH : "${runtimeLibs}" \
                  --set HF_HUB_OFFLINE "1" \
                  --set TRANSFORMERS_OFFLINE "1" \
                  --set HF_HUB_DISABLE_TELEMETRY "1" \
                  --set TRANSFORMERS_VERBOSITY "error" \
                  --run 'export TRITON_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/triton"' \
                  --run 'export TORCH_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/torch"' \
                  --run 'export HF_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"'
              fi
            done
          '';
      in
      {
        packages.default = develTools;
        packages.develTools = develTools;
        packages.runtimeEnv = runtimeEnv;

        apps.default = {
          type = "app";
          program = "${develTools}/bin/devel-nav";
        };
        apps.devel-nav = { type = "app"; program = "${develTools}/bin/devel-nav"; };
        apps.devel-watch = { type = "app"; program = "${develTools}/bin/devel-watch"; };
        apps.devel-sync = { type = "app"; program = "${develTools}/bin/devel-sync"; };
        apps.devel-search = { type = "app"; program = "${develTools}/bin/devel-search"; };
        apps.devel-tags = { type = "app"; program = "${develTools}/bin/devel-tags"; };
        apps.devel-status = { type = "app"; program = "${develTools}/bin/devel-status"; };

        # DevShell: uv manages development environment and .venv from uv.lock
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.universal-ctags
            pkgs.git
            pkgs.ruff
          ];
          env = {
            UV_PYTHON = python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
            LD_LIBRARY_PATH = runtimeLibs;
            HF_HUB_OFFLINE = "1";
            TRANSFORMERS_OFFLINE = "1";
            HF_HUB_DISABLE_TELEMETRY = "1";
            TRANSFORMERS_VERBOSITY = "error";
          };
          shellHook = ''
            unset PYTHONPATH
            export TRITON_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/triton"
            export TORCH_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/torch"
            export HF_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"
            echo "devel-tools devshell — try:  uv run pytest   |   uv run devel-search --help"
          '';
        };
      });
}
