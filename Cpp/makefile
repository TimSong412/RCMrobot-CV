CXX = g++
CXXFLAGS = -Wall $(shell pkg-config --cflags opencv4 --libs opencv4)
FILE = transform

all: transform

$(FILE)_build:
	$(CXX) $(FILE).cpp $(CXXFLAGS) -o $(FILE)

run: $(FILE)_build
	./$(FILE)