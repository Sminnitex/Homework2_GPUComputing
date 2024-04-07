CC = c++
LIB_HOME = /mnt/c/Users/Michele/source/repos/GPU
LIBS = -L$(LIB_HOME)/lib64
INCLUDE = -Isrc
OPT = -std=c++14 #-O0

#CODE
MAIN = transpose.c

######################################################################################################
BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/assignment

debug: OPT += -DDEBUG -g
debug: NVCC_FLAG += -G
debug: all

OBJECTS = $(BUILDDIR)/library.o

$(TARGETDIR)/assignment: $(MAIN) $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(LIBS) $(OPT) 

$(BUILDDIR)/library.o: src/library.c
	mkdir -p $(BUILDDIR) $(TARGETDIR)
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) src/library.c $(OPT)

clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/*
#	rm $(TARGETDIR)
