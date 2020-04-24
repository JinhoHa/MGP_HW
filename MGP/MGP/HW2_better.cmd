####################
##
## Test Condor command file
##
####################

executable	= LLtest
output		= better.out
error		= better.err
request_cpus = 16
log		= better.log
arguments	= 500 64 100000 4 1
queue
