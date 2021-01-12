{ pkgs ? import <nixpkgs> {} }:


pkgs.mkShell {
  buildInputs = with pkgs; [ nginx ] ++ stdenv.lib.optional (!stdenv.isDarwin) [ psmisc ] ;

    shellHook = ''
      cleanup() {
        kill $MDBOOK_PID
        kill $JUPYTER_PID
      }
      trap cleanup TERM EXIT QUIT

      (cd mdbook; exec mdbook serve &>/dev/null </dev/null) &
      MDBOOK_PID=$!

      JUPYTER_CONFIG=$PWD/jupyter-config.py
      nix build ../hpc-nix#jupyter -o result-jupyter
      export JUPYTER_HEADER_FILES=$PWD/include
      (cd notebooks && exec ../result-jupyter/bin/jupyter-lab --no-browser --config=$JUPYTER_CONFIG </dev/null) &
      JUPYTER_PID=$!
      echo $MDBOOK_PID $JUPYTER_PID
      nginx -c nginx/nginx.conf -p $PWD
      exit 0
    '';
}
