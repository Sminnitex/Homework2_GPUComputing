CC = nvcc
LIB_HOME = $(CURDIR)
LIBS = -L$(LIB_HOME)/lib64
INCLUDE = -Isrc
OPT = -std=c++14 -O0

#CODE
MAIN = transpose.cu

######################################################################################################
BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/assignment2

debug: OPT += -DDEBUG -g
debug: NVCC_FLAG += -G
debug: all

$(TARGETDIR)/assignment2: $(MAIN) $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ --gpu-architecture=sm_50  -o $@ $(INCLUDE) $(LIBS) $(OPT) 

clean:
	rm  $(TARGETDIR)/*
#	rm $(TARGETDIR) $(BUILDDIR)/*.o
