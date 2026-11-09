#!/bin/bash
cd /dev/shm/enc
R() { taskset -c $1 ./rga_ext "${@:2}" 2>&1 | grep -E "^fails="; }
echo "== two RGA3 in parallel (place) =="
R 5 place 0 640 360 1 3 & R 6 place 1 640 360 1 3 & wait
echo "== two processes on same RGA3_0 =="
R 5 place 0 640 360 1 3 & R 6 place 0 640 360 1 3 & wait
echo "== RGA2 draw 34 boxes 640x360 with RGA3 place in parallel =="
R 5 rect 2 640 360 34 3 & R 6 place 0 640 360 1 3 & wait
R 5 rect 2 640 360 34 3 & R 6 place 1 640 360 1 3 & wait
echo "== RGA2 draw + RGA2 place (same core) =="
R 5 rect 2 640 360 34 3 & R 6 place 2 640 360 1 3 & wait
echo "== RGA2 draw + labels on RGA3_0 =="
R 5 rect 2 640 360 34 3 & R 6 label 0 640 360 34 3 & wait
echo "== RGA2 draw + labels on RGA2 =="
R 5 rect 2 640 360 34 3 & R 6 label 2 640 360 34 3 & wait
touch rga_done.flag
