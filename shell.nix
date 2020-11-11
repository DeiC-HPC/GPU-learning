{ pkgs ? import <nixpkgs> {} }:


pkgs.mkShell {
  buildInputs = with pkgs; [ nginx ] ++ stdenv.lib.optional (!stdenv.isDarwin) [ psmisc ] ;

    shellHook = ''
      cleanup() {
        echo 'Cleaning up..'
        echo "Killing nginx ($(cat logs/nginx.pid))"
        kill $(cat logs/nginx.pid)

        echo $MDBOOK_PID $JUPYTER_PID
        kill $MDBOOK_PID
        kill $JUPYTER_PID
      }
      trap cleanup TERM EXIT QUIT

      (cd mdbook; exec mdbook serve &>/dev/null </dev/null) &
      MDBOOK_PID=$!

      JUPYTER_CONFIG=$PWD/jupyter-config.py
      (cd ../hpc-nix && nix build .#jupyter && exec ./result/bin/jupyter-lab --no-browser --config=$JUPYTER_CONFIG </dev/null) &
      JUPYTER_PID=$!
      echo $MDBOOK_PID $JUPYTER_PID
      nginx -c nginx.conf -p $PWD
    '';
}
