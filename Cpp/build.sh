g++ transform.cpp $(pkg-config --cflags opencv4 --libs opencv4) -o transform 
./transform
python show.py