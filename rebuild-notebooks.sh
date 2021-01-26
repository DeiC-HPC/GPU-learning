#!/usr/bin/env bash

set -e

SRC="$1"
DST="$2"
SOURCE_TO_NOTEBOOK="$3"

if [ -z "$SRC" ]; then
  SRC=examples
fi

if [ -z "$DST" ]; then
  DST=notebooks
fi

if [ -z "$SOURCE_TO_NOTEBOOK" ]; then
  (cd source-to-notebook && cargo build --release)
  SOURCE_TO_NOTEBOOK=$PWD/source-to-notebook/target/release/source-to-notebook
fi

rm -rf $DST || true
cp -r $SRC $DST
for f in $(find $DST -type f ); do
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
    $SOURCE_TO_NOTEBOOK $typ $f ${f%%.*}.ipynb
    rm -f $f
  else
    echo UNKNOWN $f
  fi
done
