# HW2 of MGP

## Summary

You will implement linked-list based set(LL).

The `LL` have three functions.

- `insert`
- `contains`
- `remove`

## Contents

### Expected Result

```bash
$ make submit_base
condor_submit HW2_base.cmd
Submitting job(s).
1 job(s) submitted to cluster 401.

$ make submit_better
condor_submit HW2_better.cmd
Submitting job(s).
1 job(s) submitted to cluster 402.

$ make report
cat base.out
Usage: ./a.out N_items N_threads N_test_ops extra_reads use_fine_grain
start filling
init list took 0.00211118 sec
start test
test 400000 ops took 2.67786 sec
there are 500 items left
cat better.out
Usage: ./a.out N_items N_threads N_test_ops extra_reads use_fine_grain
start filling
init list took 0.00140546 sec
start test
test 400000 ops took !@$%!@$% sec
there are 500 items left
```

### Make Functions

```bash
# Run locally
make
make run
# Compile
make LLtest
# Clean
make clean
# Submit base job to condor
make submit_base
# Submit better job to condor
make submit_better
# Check queue of condor
make queue
# Print output of outs
```

## References

- [std::thread - cplusplus.com](http://www.cplusplus.com/reference/thread/thread/)
- [OpenMP 4.5 C/C++ Syntax Guide](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf)
- [HTCondor commands cheat-sheet](https://raggleton.github.io/condor-cheatsheet/)
- [CMake Cheatsheet](https://github.com/mortennobel/CMake-Cheatsheet/blob/master/CMake_Cheatsheet.pdf)