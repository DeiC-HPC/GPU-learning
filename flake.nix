{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hpc-nix = {
      url = "/home/tethys/git/hpc-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, hpc-nix, ... }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      packages."${system}" = rec {
        mdbook-classy = pkgs.callPackage ./nix/mdbook-classy.nix { };
        mdbook-content = pkgs.stdenv.mkDerivation {
          name = "mdbook-content";
          phases = [ "installPhase" ];
          buildInputs = [ pkgs.mdbook mdbook-classy ];
          installPhase = ''
            cd ${./.}/mdbook
            mdbook build -d $out
          '';
        };
        docker-nginx-conf = pkgs.stdenv.mkDerivation {
          name = "docker-nginx-conf";
          src = ./nginx;
          phases = [ "installPhase" ];
          installPhase = ''
            mkdir $out
            cp $src/proxy.conf $src/mime.types $out
            export book=${mdbook-content}/
            substituteAll $src/docker.conf $out/nginx.conf
          '';
        };
        source-to-notebook = pkgs.rustPlatform.buildRustPackage {
          name = "source-to-notebook";
          src = ./source-to-notebook;
          cargoSha256 = "01xbi24cq065si654d337x4l22zwxvnpx2kl5n2pq6dpavmbxw59";
        };
        notebooks = pkgs.stdenv.mkDerivation {
          name = "notebooks";
          src = ./.;
          phases = [ "unpackPhase" "installPhase" ];
          buildInputs = [ source-to-notebook pkgs.bash ];
          installPhase = ''
            patchShebangs ./rebuild-notebooks.sh
            ./rebuild-notebooks.sh ./examples $out source-to-notebook
          '';
        };
        jupyter = pkgs.writeScript "jupyter" ''
          #!${pkgs.bash}/bin/bash

          set -e

          TMPDIR=$(mktemp -d /tmp/jupyter-notebooks.XXXXXX)
          trap "rm -rf $TMPDIR" EXIT INT QUIT TERM
          cp -r ${notebooks} $TMPDIR/notebooks
          chmod u+w -R $TMPDIR
          cd $TMPDIR/notebooks

          export JUPYTER_HEADER_FILES=${./include}
          ${hpc-nix.packages."${system}".jupyter}/bin/jupyter-lab --no-browser --config=${./jupyter-config.py} </dev/null
        '';
        docker-nginx-command = pkgs.writeScript "nginx" ''
          #!${pkgs.bash}/bin/bash

          set -e

          ${jupyter} &
          JUPYTER_PID=$!
          trap 'kill $JUPYTER_PID' TERM EXIT QUIT
          ${pkgs.nginx}/bin/nginx -c ${docker-nginx-conf}/nginx.conf -p $PWD
        '';
        docker-nginx = pkgs.dockerTools.buildImage {
          name = "GPU-learning";
          config.cmd = ["${docker-nginx-command}"];
        };
      };
    };
}
