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
        docker-nginx-command = pkgs.writeScript "docker-nginx-command" ''
          #!${pkgs.bash}/bin/bash

          set -e

          ${jupyter} &
          JUPYTER_PID=$!
          trap 'kill $JUPYTER_PID' TERM EXIT QUIT
          cd /tmp
          mkdir logs
          ${pkgs.nginx}/bin/nginx -c ${docker-nginx-conf}/nginx.conf -p $PWD
        '';
        docker-nginx = pkgs.dockerTools.buildImage {
          name = "GPU-learning";
          config = {
            Env = [ "PATH=${pkgs.coreutils}/bin" ];
            Cmd = ["${docker-nginx-command}"];
            User = 1000;
            Group = 100;
          };
          extraCommands = ''
            mkdir -m 0777 tmp var/cache/nginx
            mkdir -p etc bin usr/bin
            ln -s ${pkgs.bash}/bin/sh bin
            ln -s ${pkgs.coreutils}/bin/env usr/bin
            echo 'root:x:0:0:root:/root:/bin/sh' > etc/passwd
            echo 'user:x:1000:100::/:/bin/sh' >> etc/passwd
            echo 'nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin' >> etc/passwd
            echo 'root:x:0:' > etc/group
            echo 'users:x:100:' >> etc/group
            echo 'nogroup:x:65534:' >> etc/group
          '';
        };
      };
    };
}
