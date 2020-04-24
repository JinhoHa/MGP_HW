####################
##
## Test Condor command file
##
####################

executable	= LLtest
output		= base.out
error		= base.err
request_cpus = 16
log		= base.log
arguments	= 500 64 100000 4 0
queue
