#!/bin/bash
# The stem-110 engine run of full/cof1 with the modules' exact d2 from module_d2.
cd $CLAUDE_JOB_DIR/tmp
B=/wsu/home/hd/hd72/hd7264/Documents/sseq/.claude/worktrees/derivations-global/ext/target/release/derivations
R=/rs/rs_grp_csht/resolutions/sphere_gpu_n400/ring_n400_nsweep
/usr/bin/time -v nice -n 19 $B $R 110 70 --gens-only --diagram md110/free.dg --from-e2 --pages 4 --secondary secprod/easy3_n200.out --check 3:full/hold_d3.csv --check 4:full/hold_d4.csv > md110/free.out 2> md110/free.log
