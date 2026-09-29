#!/bin/sh
# usage: run26.sh tag window rainfile extra...
tag=$1; w=$2; rf=$3; shift 3
cd ..
target/release/examples/flood_demo --gis ../windows/cells_w$w.csv --rain ../windows/$rf --cell-m 100 --epochs 1 --seed 1 --out ../results/${tag}_w$w --json ../results/${tag}_w$w.json "$@" > ../results/${tag}_w$w.log 2>&1
