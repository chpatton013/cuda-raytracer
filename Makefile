SRC_DIR = src
OBJ_DIR = obj
LIBS = m
INCS = $(SRC_DIR) third_party/tclap/include
SRCS = $(shell find $(SRC_DIR) -name "*.cc")
SRCS_CUDA = $(shell find $(SRC_DIR) -name "*.cu")
DEPS = $(shell find $(SRC_DIR) -name "*.h")
DEPS_CUDA = $(shell find $(SRC_DIR) -name "*.cuh")
OBJS = $(patsubst $(SRC_DIR)/%.o,$(OBJ_DIR)/%.o,$(SRCS:.cc=.o))
OBJS_CUDA = $(patsubst $(SRC_DIR)/%.o,$(OBJ_DIR)/%.o,$(SRCS_CUDA:.cu=.o))
SRC_SUB_DIRS = $(shell find $(SRC_DIR) -type d)
OBJ_SUB_DIRS = $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(SRC_SUB_DIRS))
EXEC = $(shell basename `pwd`)


#CXX = g++
#CFLAGS = -Wall -Wextra -pipe -DGL_GLEXT_PROTOTYPES $(foreach d,$(INCS),-I$d)

CXX = nvcc
CFLAGS=-g -c -arch=compute_20 -code=sm_20 -DGL_GLEXT_PROTOTYPES $(foreach d,$(INCS),-I$d)
LD = nvcc
LDFLAGS =


.PHONY: all debug test release profile prepare clean remove

all test: debug
debug: CFLAGS += #-g3-DDEBUG
release: CFLAGS += -Ofast -DNDEBUG
profile: CFLAGS += -g3 -pg -Og -DNDEBUG
profile: LDFLAGS += -pg


debug release profile: $(EXEC)

test: $(EXEC)
	./$(EXEC)

prepare:
	mkdir -p $(OBJ_SUB_DIRS)

clean:
	rm -f $(OBJS)
	rm -f $(OBJS_CUDA)	
	rm -f output.tga
	rm -f ray
	rm -rf $(OBJ_SUB_DIRS)

remove: clean
	rm -f $(EXEC)

$(EXEC): prepare $(OBJS) $(OBJS_CUDA)
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(OBJS_CUDA) $(foreach l,$(LIBS),-l$l)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(DEPS)
	$(CXX) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CXX) $(CFLAGS) -c -o $@ $<
	
	