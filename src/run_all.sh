#!/usr/bin/env bash
# Run all experimental configurations sequentially; skip finished ones.
cd "$(dirname "$0")"
run() {  # purity seed [--attacks]
  tag=$(printf "p%.2f_s%d" "$1" "$2")
  [ -f "../results/runs/$tag.json" ] && { echo "skip $tag"; return; }
  python -W ignore 02_run.py --purity "$1" --seed "$2" $3 2>&1 | grep --line-buffered -v -i -E "deprecat|iscoroutine" > "../results/logs/$tag.log"
  tail -1 "../results/logs/$tag.log"
}
mkdir -p ../results/logs
for s in 0 1 2 3 4; do run 0.85 $s --attacks; done
for s in 0 1 2 3 4; do run 0.00 $s; done
for p in 0.50 1.00; do for s in 0 1 2; do run $p $s; done; done
# sensitivity analysis: extended feature sets (01b_build_ext_features.py) on the same partitions and splits
runx() {  # features purity seed
  tag=$(printf "%s_p%.2f_s%d" "$1" "$2" "$3")
  [ -f "../results/runs/$tag.json" ] && { echo "skip $tag"; return; }
  python -W ignore 02_run.py --features "$1" --purity "$2" --seed "$3" 2>&1 | grep --line-buffered -v -i -E "deprecat|iscoroutine" > "../results/logs/$tag.log"
  tail -1 "../results/logs/$tag.log"
}
[ -f ../data/ext_icu_rx_micro.parquet ] && for s in 0 1 2 3 4; do runx ext 0.85 $s; done
[ -f ../data/ext_chart.parquet ] && for s in 0 1 2 3 4; do runx ext2 0.85 $s; done
# revision analyses (2026-11): data-scarce sites (B1) and natural care-era partition (B4)
runr() {  # tag-prefix purity seed extra-args...
  pre=$1; p=$2; s=$3; shift 3
  tag=$(printf "%sp%.2f_s%d" "$pre" "$p" "$s")
  [ -f "../results/runs/$tag.json" ] && { echo "skip $tag"; return; }
  python -W ignore 02_run.py --purity "$p" --seed "$s" "$@" 2>&1 | grep --line-buffered -v -i -E "deprecat|iscoroutine" > "../results/logs/$tag.log"
  tail -1 "../results/logs/$tag.log"
}
for f in 0.05 0.1 0.2; do for s in 0 1 2 3 4; do runr "small${f}_" 0.85 $s --frac $f; done; done
for s in 0 1 2 3 4; do runr "year_" 0.85 $s --partition year; done
# stealthier attacks (B2), calibration fixes (B3), federated LR (B5): 02c_extensions.py
for s in 0 1 2 3 4; do
  tag="ext3_p0.85_s$s"
  [ -f "../results/runs/$tag.json" ] && { echo "skip $tag"; continue; }
  python -W ignore 02c_extensions.py --seed $s 2>&1 | grep --line-buffered -v -i -E "deprecat|iscoroutine" > "../results/logs/$tag.log"
  tail -1 "../results/logs/$tag.log"
done
echo ALL_DONE
