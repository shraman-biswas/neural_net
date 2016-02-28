CXX	=	gcc
FLAGS	=	-Wall -O3 -ggdb
LIBS	=	-lgsl -lgslcblas -lm
SOURCE	=	main.c neural_net.c
BIN	=	main

all:
	$(CXX) $(FLAGS) $(SOURCE) -o $(BIN) $(LIBS)

clean:
	rm -f *~ $(BIN)
