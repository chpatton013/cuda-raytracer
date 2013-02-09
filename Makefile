SRC_DIR = src
OBJ_DIR = obj

LIBS = m GL GLU glut
INCS = $(SRC_DIR) third_party/tclap/include
DEFS = GL_GLEXT_PROTOTYPES

SRCS = $(shell find $(SRC_DIR) -name "*.cu")
DEPS = $(shell find $(SRC_DIR) -name "*.h")
OBJS = $(patsubst $(SRC_DIR)/%.o,$(OBJ_DIR)/%.o,$(SRCS:.cu=.o))
SRC_SUB_DIRS = $(shell find $(SRC_DIR) -type d)
OBJ_SUB_DIRS = $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(SRC_SUB_DIRS))
EXEC = $(shell basename `pwd`)

CXX = nvcc
CFLAGS = -arch=compute_20 -code=sm_21 -Xptxas -dlcm=ca $(foreach d,$(DEFS),-D$d) $(foreach d,$(INCS),-I$d)
LD = nvcc
LDFLAGS =

.PHONY: all debug run release profile prepare clean remove

all: debug
run: all
debug: CFLAGS += -g -DDEBUG
release: CFLAGS += -O3 -DNDEBUG
profile: CFLAGS += -g -pg -O3 -DNDEBUG
profile: LDFLAGS += -pg

debug release profile: $(EXEC)

run: $(EXEC)
	./$(EXEC)

prepare:
	mkdir -p $(OBJ_SUB_DIRS)

clean:
	rm -f $(OBJS)
	rm -rf $(OBJ_SUB_DIRS)

remove: clean
	rm -f $(EXEC)

$(EXEC): prepare $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(foreach l,$(LIBS),-l$l)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CXX) $(CFLAGS) -c -o $@ $<
