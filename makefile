CC = clang
CFLAGS = -g -Wall -O2 -c
LDFLAGS = -lgsl -lgslcblas -lm -o

CFILES = *.c
OFILES = *.o
OUTPUT = main

CXX	=	gcc
FLAGS	=	-Wall -O3 -ggdb
LIBS	=	-lgsl -lgslcblas -lm
SOURCE	=	main.c neural_net.c
BIN	=	main

all:
	$(CXX) $(FLAGS) $(SOURCE) -o $(BIN) $(LIBS)

clean:
	rm -f *~ $(BIN)
