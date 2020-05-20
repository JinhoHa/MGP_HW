############################################
##
## Join Tables Condor command file
##
############################################

executable	 = jointable
output		 = result/jointable.out
error		 = result/jointable.err
log		     = result/jointable.log
request_cpus = 16
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = data/input_10m10m.txt, data/output_10m10m.txt
arguments	            = input_10m10m.txt output_10m10m.txt 0
queue