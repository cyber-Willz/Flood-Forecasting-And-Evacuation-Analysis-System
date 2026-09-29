#!/bin/sh
# usage: runv.sh tag window extra-args...
tag=$1; w=$2; shift 2
cd ..
target/release/examples/flood_demo --gis ../windows/cells_w$w.csv --rain ../windows/rain.csv --cell-m 100 --epochs 1 --seed 1 --out ../results/${tag}_w$w --json ../results/${tag}_w$w.json "$@" > ../results/${tag}_w$w.log 2>&1
