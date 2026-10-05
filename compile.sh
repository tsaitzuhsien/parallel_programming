#!/bin/bash

name="$1"

clang "${name}.c" -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp -o "$name"
