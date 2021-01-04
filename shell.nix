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

      rm -rf notebooks
      cp -r examples notebooks
      for f in $(find notebooks -type f ); do
        ext="''${f##*.}"
        typ=""
        case $ext in
          py) typ=python;;
          cpp)
            if echo $f | grep -q openacc; then
              typ=cpp_openacc
            else
              typ=cpp_openmp
            fi
          ;;
          f90)
            if echo $f | grep -q openacc; then
              typ=fortran_openacc
            else
              typ=fortran_openmp
            fi
          ;;
          cu) typ=cuda;;
        esac
        if [[ -n "$typ" ]]; then
          (cd source-to-notebook && cargo run $typ ../$f ../''${f%%.*}.ipynb)
          rm -f $f
        else
          echo UNKNOWN $f
        fi
      done

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
