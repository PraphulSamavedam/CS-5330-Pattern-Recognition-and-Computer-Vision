# MacOS compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

# Make it the C++ compiler too
CXX = $(CC)

# OSX include paths for MacPorts
CFLAGS = -Wc++11-extensions -std=c++11 -I/opt/homebrew/Cellar/opencv/4.7.0/include/opencv4 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to wwhere your openCV libraries are stored
# these settings work for macports
#LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.7.0/include/opencv4/ -L/opt/homebrew/Cellar/opencv/4.7.0/lib/
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.7.0/lib/# opencv libraries are here

# opencv libraries and MacOS frameworks
# these settings work for macOS and macports
LDLIBS = -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_calib3d

# Location to write the executables
BINDIR = .

# rules for making various executables
all: main project features

main: ./src/main.o ./src/tasks.o ./src/csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	
project: ./src/project.o ./src/tasks.o ./src/csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	
features: ./src/features.o ./src/tasks.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	
clean:
	rm -f *.o *~


