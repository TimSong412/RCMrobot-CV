g++ imgcut.cpp $(pkg-config --cflags opencv4 --libs opencv4) -o imgcut
./imgcut