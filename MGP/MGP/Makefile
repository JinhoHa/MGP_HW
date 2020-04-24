# LLtest
run: LLtest
	./LLtest 500 64 100000 4 0
	./LLtest 500 64 100000 4 1

LLtest: main.cc 
	g++ -std=c++11 -g -pthread  $< -o $@ -lboost_system  -lboost_thread

clean:
	rm -f LLtest *.err *.log *.out

# Condor
submit_base: LLtest
	condor_submit HW2_base.cmd

submit_better: LLtest
	condor_submit HW2_better.cmd

report:
	cat base.out
	cat better.out

queue:
	condor_q

status:
	condor_status

remove:
	condor_rm

# Development
format:
	clang-format -i -style=Google *.cc *.h
