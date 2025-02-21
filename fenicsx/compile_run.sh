gfortran -c umat.f -o umatf.o
gcc -c umat.c -o umatc.o
gcc -shared umatc.o umatf.o -o umat.so -lgfortran
python3 main.py