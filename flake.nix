{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hpc-nix = {
      url = "github:DeiC-HPC/gpu-jupyter";
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
      pythonenv = pkgs.python3.withPackages(ps: with ps; [ mkdocs mkdocs-material ]);
    in rec {
      packages."${system}" = rec {
        mkdocs-content = pkgs.stdenv.mkDerivation {
          name = "mkdocs-content";
          phases = [ "installPhase" ];
          buildInputs = [ pythonenv ];
          installPhase = ''
            cd ${./.}/book
            mkdocs build -d $out
          '';
        };
        docker-nginx-conf = pkgs.stdenv.mkDerivation {
          name = "docker-nginx-conf";
          src = ./nginx;
          phases = [ "installPhase" ];
          installPhase = ''
            mkdir $out
            cp $src/proxy.conf $src/mime.types $out
            export book=${mkdocs-content}/
            substituteAll $src/docker.conf $out/nginx.conf
          '';
        };
        source-to-notebook = pkgs.rustPlatform.buildRustPackage {
          name = "source-to-notebook";
          src = ./source-to-notebook;
          #cargoSha256 = "01xbi24cq065si654d337x4l22zwxvnpx2kl5n2pq6dpavmbxw59";
          cargoHash = "sha256-DA3WpN8LblMgjWH9IXb7mA1N4tzNi9y1Kp7kCu4XqLE=";
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
          ${hpc-nix.packages."${system}".jupyterlab}/bin/jupyter-lab --no-browser -y --config=${./jupyter-config.py} --ip=0.0.0.0 </dev/null
        '';
        docker-nginx-command = pkgs.writeScript "docker-nginx-command" ''
          #!${pkgs.bash}/bin/bash

          set -e

          ${jupyter} &
          JUPYTER_PID=$!
          echo $JUPYTER_PID
          #trap 'kill $JUPYTER_PID' TERM EXIT QUIT INT
          cd /tmp
          mkdir -p logs
          ${pkgs.nginx}/bin/nginx -c ${docker-nginx-conf}/nginx.conf -p $PWD
        '';
        docker-nginx = pkgs.dockerTools.buildImage {
          name = "GPU-learning";
          config = {
            Env = [ "PATH=${pkgs.coreutils}/bin:${pkgs.gnused}/bin:${pkgs.curl}/bin:${pkgs.vim}/bin" ]; Cmd = ["${docker-nginx-command}"];
            User = "1000";
            Group = "100";
          };
          extraCommands = ''
            mkdir -p -m 0777 tmp var/cache/nginx
            mkdir -p etc bin usr/bin
            ln -s ${pkgs.bash}/bin/sh bin
            ln -s ${pkgs.coreutils}/bin/env usr/bin
            echo 'root:x:0:0:root:/tmp:/bin/sh' > etc/passwd
            echo 'user:x:1000:100::/tmp:/bin/sh' >> etc/passwd
            echo 'nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin' >> etc/passwd
            echo 'root:x:0:' > etc/group
            echo 'users:x:100:' >> etc/group
            echo 'nogroup:x:65534:' >> etc/group
          '';
        };
        singularity-nginx = pkgs.singularity-tools.buildImage {
          name = "GPU-learning";
          runScript = "${docker-nginx-command}";
          diskSize = 100000;
          memSize = 50000;
          contents = [
            pkgs.coreutils
            pkgs.gnused
            pkgs.curl
            pkgs.vim
            pkgs.ncurses
          ];
          # config = {
          #   Env = [ "PATH=${pkgs.coreutils}/bin:${pkgs.gnused}/bin:${pkgs.curl}/bin:${pkgs.vim}/bin" ]; Cmd = ["${docker-nginx-command}"];
          #   User = "1000";
          #   Group = "100";
          # };
          runAsRoot = ''
            mkdir -p -m 0777 tmp var/cache/nginx
            mkdir -p etc bin usr/bin
            ln -s ${pkgs.bash}/bin/sh bin
            ln -s ${pkgs.coreutils}/bin/env usr/bin
            echo 'root:x:0:0:root:/tmp:/bin/sh' > etc/passwd
            echo 'user:x:1000:100::/tmp:/bin/sh' >> etc/passwd
            echo 'nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin' >> etc/passwd
            echo 'root:x:0:' > etc/group
            echo 'users:x:100:' >> etc/group
            echo 'nogroup:x:65534:' >> etc/group
          '';
        };
      };
      devShell."${system}" = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [ pythonenv ];
      };
    };
}
