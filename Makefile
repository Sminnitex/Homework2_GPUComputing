CC = c++
LIB_HOME = /mnt/c/Users/Michele/source/repos/GPU
LIBS = -L$(LIB_HOME)/lib64
INCLUDE = -Isrc
OPT = -std=c++14 -O3

#CODE
MAIN = gemmblock.c

######################################################################################################
BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/lab2part2

debug: OPT += -DDEBUG -g
debug: NVCC_FLAG += -G
debug: all

OBJECTS = $(BUILDDIR)/library.o

$(TARGETDIR)/lab2part2: $(MAIN) $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(LIBS) $(OPT) 

$(BUILDDIR)/library.o: src/library.c
	mkdir -p $(BUILDDIR) $(TARGETDIR)
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) src/library.c $(OPT)

clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/*
#	rm $(TARGETDIR)
