# define the C compiler to use
CC = g++
CU = nvcc
CFLAGS = -Wall -g
CUFLAGS =  -g -arch=compute_35 -code=sm_35 -lcudadevrt -rdc=true
INCLUDES = -I includes
CPPFLAGS = $(INCLUDES)

# define library paths in addition to /usr/lib
LFLAGS =

# define any libraries to link into executable:
LIBS =

# build directory
BUILD=build
SRCDIR=src
BIN=$(BUILD)/bin
OBJECTS=$(BUILD)/objects

# output dir
OUTPUT=output

# define the C source files
SRCS_CC = $(wildcard $(SRCDIR)/*.cpp)
SRCS_CU = $(wildcard $(SRCDIR)/*.cu)

# define the C object files
OBJS_CC := $(patsubst $(SRCDIR)/%.cpp, $(OBJECTS)/%.o, $(SRCS_CC) )

# define the CU object files
OBJS_CU := $(patsubst $(SRCDIR)/%.cu, $(OBJECTS)/%.o, $(SRCS_CU) )

# define the executable file
MAIN := ${BIN}/digitalFilter

.PHONY: depend clean

all: $(MAIN)
	@echo  Simple compiler named `basename $(MAIN)`  has been compiled

$(MAIN): $(OBJS_CU) $(OBJS_CC)
	mkdir -p ${BIN} ${OUTPUT}
	@echo $(OBJECTS)
	$(CU) $(CUFLAGS) $(CPPFLAGS) -o $(MAIN) $(OBJS_CU) $(OBJS_CC) $(LFLAGS) $(LIBS)
	cp ${MAIN} .


# Compile all the .c into .o's
$(OBJS_CC) : $(OBJECTS)/%.o : $(SRCDIR)/%.cpp
	mkdir -p ${OBJECTS}
	$(CU) -dc  $(CUFLAGS) $(CPPFLAGS) $< -o $@

# Compile all the .cu into .o's
$(OBJS_CU) : $(OBJECTS)/%.o : $(SRCDIR)/%.cu
	mkdir -p ${OBJECTS}
	$(CU) -dc  $(CUFLAGS) $(CPPFLAGS) $< -o $@

$(SRCDIR)/%cu : $(SRCDIR)/%.h
	@echo --checked

clean:
	@echo cleaning
	$(RM)  */*.o digitalFil*  $(OBJECTS)/* ${OUTPUT}/*  *~ $(MAIN)

depend: $(SRCS_CC)
	@echo $(INCLUDES)
	makedepend $(INCLUDES) $^

# DO NOT DELETE
