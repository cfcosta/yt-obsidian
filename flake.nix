{
  description = "hello world application using uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      cudaOverlay = pkgs: final: prev: {
        flash-attn = prev.flash-attn.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
          ];

          buildInputs = (old.buildInputs or [ ]) ++ [
            (final.resolveBuildSystem {
              setuptools = [ ];
              torch = [ ];
              psutil = [ ];
            })
          ];

          CUDA_HOME = pkgs.symlinkJoin {
            name = "cuda-redist";
            paths = with pkgs.cudaPackages; [
              cuda_cudart
              cuda_nvcc
              pkgs.cudatoolkit
            ];
          };
        });

        numba = prev.numba.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.onetbb
          ];
        });

        nvidia-cufile-cu12 = prev.nvidia-cufile-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.rdma-core ];
        });

        nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudatoolkit
            pkgs.linuxPackages.nvidia_x11
          ];
        });

        nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudatoolkit
            pkgs.linuxPackages.nvidia_x11
          ];
        });

        nvidia-nvshmem-cu12 = prev.nvidia-nvshmem-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.rdma-core
            pkgs.libGL
            pkgs.libGLU
            pkgs.pmix
            pkgs.libfabric
            pkgs.mpi
          ];
        });

        soundfile = prev.soundfile.overrideAttrs (_: {
          postInstall = ''
            substituteInPlace $out/lib/python*/site-packages/soundfile.py --replace "_find_library('sndfile')" "'${pkgs.libsndfile.out}/lib/libsndfile${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}'"
          '';
        });

        sox = prev.sox.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
            (final.resolveBuildSystem { setuptools = [ ]; })
          ];
        });

        torch = prev.torch.overrideAttrs (_: {
          buildInputs = pkgs.python3Packages.torchWithCuda.buildInputs ++ [ pkgs.cudaPackages.libnvshmem ];

          postFixup = ''
            addAutoPatchelfSearchPath "${final.nvidia-cusparselt-cu12}"
          '';

          autoPatchelfIgnoreMissingDeps = [
            "libcuda.so.1" # this will be found at runtime?
            "libnvjitlink.so"
            "libnvidia-ml.so.1"
          ];
        });

        torchaudio =
          let
            FFMPEG_ROOT = pkgs.symlinkJoin {
              name = "ffmpeg";
              paths = with pkgs; [
                ffmpeg_6-full.bin
                ffmpeg_6-full.dev
                ffmpeg_6-full.lib
              ];
            };
          in
          prev.torchaudio.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.sox
              pkgs.libgccjit
            ];
            inherit FFMPEG_ROOT;
            autoPatchelfIgnoreMissingDeps = true;
            postFixup = ''
              addAutoPatchelfSearchPath "${final.torch}/${pkgs.python3.sitePackages}/torch/lib"
            '';
          });
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          python = pkgs.python3;
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              overlay
              pyproject-build-systems.overlays.wheel
              (cudaOverlay pkgs)
            ]
          )
      );

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          pythonSet = pythonSets.${system}.overrideScope (
            lib.composeManyExtensions [
              editableOverlay
              (cudaOverlay pkgs)
            ]
          );
          virtualenv = pythonSet.mkVirtualEnv "yt-obsidian-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.sox
              pkgs.libgccjit
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
              LD_LIBRARY_PATH = "/usr/lib64:/usr/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "yt-obsidian-env" workspace.deps.default;
      });
    };
}
