CC=g++
CXXFLAGS=-O3 `pkg-config --cflags opencv` -Iinclude
LDFLAGS=`pkg-config --libs opencv`


.PHONY: all clean
IntensityOrderFeature: src/Main.cpp MyDescriptors.o Utils.o
	$(CC) $(CXXFLAGS) -o $@ MyDescriptors.o Utils.o src/Main.cpp $(LDFLAGS)

MyDescriptors.o: include/Common.h Utils.o include/MyDescriptors.h src/MyDescriptors.cpp
	$(CC) $(CXXFLAGS) -c src/MyDescriptors.cpp -o $@

Utils.o: include/Common.h include/Utils.h src/Utils.cpp
	$(CC) $(CXXFLAGS) -c src/Utils.cpp -o $@	

all: IntensityOrderFeature

clean:
	rm -f *.o IntensityOrderFeature

