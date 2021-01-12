#!/usr/bin/env bash

set -e

rm -rf notebooks
cp -r examples notebooks
for f in $(find notebooks -type f ); do
  ext="${f##*.}"
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
    (cd source-to-notebook && cargo run --release $typ ../$f ../${f%%.*}.ipynb)
    rm -f $f
  else
    echo UNKNOWN $f
  fi
done
