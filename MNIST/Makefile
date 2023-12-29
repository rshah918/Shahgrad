# Makefile for load_data.cpp

# Compiler
CXX := g++
# Compiler flags
CXXFLAGS := -std=c++11
# Source file
SOURCE := load_data.cpp
# Output executable
OUTPUT := a.out

# Libraries and flags for OpenCV
OPENCV_LIBS := $(shell pkg-config --cflags --libs opencv4)

# Build rule
all: $(OUTPUT)

$(OUTPUT): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(OUTPUT) $(OPENCV_LIBS)

# Clean rule
clean:
	rm -f $(OUTPUT)
