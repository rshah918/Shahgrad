# Makefile for train_mnist.cpp

# Compiler
CXX := g++
# Compiler flags
CXXFLAGS := -std=c++11
# Source file
SOURCE1 := train_mnist.cpp 
SOURCE2 := ../shahgrad.cpp
# Output executable
OUTPUT := a.out

# Libraries and flags for OpenCV
OPENCV_LIBS := $(shell pkg-config --cflags --libs opencv4)

# Build rule
all: $(OUTPUT)

$(OUTPUT): $(SOURCE1) $(SOURCE2)
	$(CXX) $(CXXFLAGS) $(SOURCE1) -o $(OUTPUT) $(OPENCV_LIBS)

# Clean rule
clean:
	rm -f $(OUTPUT)
