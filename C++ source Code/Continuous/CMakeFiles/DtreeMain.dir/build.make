# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous"

# Include any dependencies generated for this target.
include CMakeFiles/DtreeMain.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DtreeMain.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DtreeMain.dir/flags.make

CMakeFiles/DtreeMain.dir/main.o: CMakeFiles/DtreeMain.dir/flags.make
CMakeFiles/DtreeMain.dir/main.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DtreeMain.dir/main.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DtreeMain.dir/main.o -c "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/main.cpp"

CMakeFiles/DtreeMain.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DtreeMain.dir/main.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/main.cpp" > CMakeFiles/DtreeMain.dir/main.i

CMakeFiles/DtreeMain.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DtreeMain.dir/main.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/main.cpp" -o CMakeFiles/DtreeMain.dir/main.s

CMakeFiles/DtreeMain.dir/main.o.requires:

.PHONY : CMakeFiles/DtreeMain.dir/main.o.requires

CMakeFiles/DtreeMain.dir/main.o.provides: CMakeFiles/DtreeMain.dir/main.o.requires
	$(MAKE) -f CMakeFiles/DtreeMain.dir/build.make CMakeFiles/DtreeMain.dir/main.o.provides.build
.PHONY : CMakeFiles/DtreeMain.dir/main.o.provides

CMakeFiles/DtreeMain.dir/main.o.provides.build: CMakeFiles/DtreeMain.dir/main.o


CMakeFiles/DtreeMain.dir/CDTree.o: CMakeFiles/DtreeMain.dir/flags.make
CMakeFiles/DtreeMain.dir/CDTree.o: CDTree.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/DtreeMain.dir/CDTree.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DtreeMain.dir/CDTree.o -c "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/CDTree.cpp"

CMakeFiles/DtreeMain.dir/CDTree.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DtreeMain.dir/CDTree.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/CDTree.cpp" > CMakeFiles/DtreeMain.dir/CDTree.i

CMakeFiles/DtreeMain.dir/CDTree.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DtreeMain.dir/CDTree.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/CDTree.cpp" -o CMakeFiles/DtreeMain.dir/CDTree.s

CMakeFiles/DtreeMain.dir/CDTree.o.requires:

.PHONY : CMakeFiles/DtreeMain.dir/CDTree.o.requires

CMakeFiles/DtreeMain.dir/CDTree.o.provides: CMakeFiles/DtreeMain.dir/CDTree.o.requires
	$(MAKE) -f CMakeFiles/DtreeMain.dir/build.make CMakeFiles/DtreeMain.dir/CDTree.o.provides.build
.PHONY : CMakeFiles/DtreeMain.dir/CDTree.o.provides

CMakeFiles/DtreeMain.dir/CDTree.o.provides.build: CMakeFiles/DtreeMain.dir/CDTree.o


# Object files for target DtreeMain
DtreeMain_OBJECTS = \
"CMakeFiles/DtreeMain.dir/main.o" \
"CMakeFiles/DtreeMain.dir/CDTree.o"

# External object files for target DtreeMain
DtreeMain_EXTERNAL_OBJECTS =

DtreeMain: CMakeFiles/DtreeMain.dir/main.o
DtreeMain: CMakeFiles/DtreeMain.dir/CDTree.o
DtreeMain: CMakeFiles/DtreeMain.dir/build.make
DtreeMain: CMakeFiles/DtreeMain.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable DtreeMain"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DtreeMain.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DtreeMain.dir/build: DtreeMain

.PHONY : CMakeFiles/DtreeMain.dir/build

CMakeFiles/DtreeMain.dir/requires: CMakeFiles/DtreeMain.dir/main.o.requires
CMakeFiles/DtreeMain.dir/requires: CMakeFiles/DtreeMain.dir/CDTree.o.requires

.PHONY : CMakeFiles/DtreeMain.dir/requires

CMakeFiles/DtreeMain.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DtreeMain.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DtreeMain.dir/clean

CMakeFiles/DtreeMain.dir/depend:
	cd "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous" "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous" "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous" "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous" "/home/liana/Desktop/MachineLearning/MachineLearning-DecisionTree/C++ source Code/Continuous/CMakeFiles/DtreeMain.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/DtreeMain.dir/depend

