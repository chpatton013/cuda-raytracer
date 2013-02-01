SRC_DIR = src
OBJ_DIR = obj

LIBS = m
INCS = $(SRC_DIR) third_party/tclap/include

SRCS = $(shell find $(SRC_DIR) -name "*.cu")
DEPS = $(shell find $(SRC_DIR) -name "*.h")
OBJS = $(patsubst $(SRC_DIR)/%.o,$(OBJ_DIR)/%.o,$(SRCS:.cu=.o))
SRC_SUB_DIRS = $(shell find $(SRC_DIR) -type d)
OBJ_SUB_DIRS = $(patsubst $(SRC_DIR)%,$(OBJ_DIR)%,$(SRC_SUB_DIRS))
EXEC = $(shell basename `pwd`)


CXX = nvcc
CFLAGS = -arch=compute_20 -code=sm_20 $(foreach d,$(INCS),-I$d)
LD = nvcc
LDFLAGS =


.PHONY: all debug test release profile prepare clean remove

all: debug
run: all
debug: CFLAGS += -g -DDEBUG
release: CFLAGS += -O2 -use-fast-math -DNDEBUG
profile: CFLAGS += -g -pg -O2 -use-fast-math -DNDEBUG
profile: LDFLAGS += -pg

debug release profile: $(EXEC)

test: $(EXEC)
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
