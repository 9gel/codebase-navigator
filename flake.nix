{
  description = "codebase-navigator — Git-aware ctags indexing, live watchers, and LanceDB semantic search";

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

        # Disable auto-patchelf for torch wheel so its $ORIGIN RPATHs are preserved
        pyprojectOverrides = final: prev: {
          torch = prev.torch.overrideAttrs (_: { dontAutoPatchelf = true; });
        };

        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          lockedOverlay
          pyprojectOverrides
        ]);

        runtimeEnv = pythonSet.mkVirtualEnv "codebase-navigator-env" workspace.deps.default;
        runtimeLibs = "${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}";

        # Runtime wrapper with universal-ctags and git (online by default so the
        # embedding model can be downloaded on first run; cached afterwards).
        codebaseNavigator = pkgs.runCommand "codebase-navigator"
          { nativeBuildInputs = [ pkgs.makeWrapper ]; }
          ''
            mkdir -p $out/bin
            for b in cn; do
              if [ -f "${runtimeEnv}/bin/$b" ]; then
                makeWrapper "${runtimeEnv}/bin/$b" "$out/bin/$b" \
                  --prefix PATH : "${lib.makeBinPath [ pkgs.universal-ctags pkgs.git pkgs.coreutils ]}" \
                  --prefix LD_LIBRARY_PATH : "${runtimeLibs}" \
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
        packages.default = codebaseNavigator;
        packages.codebaseNavigator = codebaseNavigator;
        packages.runtimeEnv = runtimeEnv;

        apps.default = {
          type = "app";
          program = "${codebaseNavigator}/bin/cn";
        };
        apps.cn = { type = "app"; program = "${codebaseNavigator}/bin/cn"; };

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
            HF_HUB_DISABLE_TELEMETRY = "1";
            TRANSFORMERS_VERBOSITY = "error";
          };
          shellHook = ''
            unset PYTHONPATH
            export TRITON_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/triton"
            export TORCH_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/torch"
            export HF_HOME="''${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"
            echo "codebase-navigator devshell — try:  uv run pytest   |   uv run cn --help"
          '';
        };
      });
}
