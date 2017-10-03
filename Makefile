kadai1: kadai1.c kadai1.sub
	mpicc -o kadai1 kadai1.c
	condor_submit kadai1.sub

clean:
	rm kadai1.err kadai1.out kadai1.log
