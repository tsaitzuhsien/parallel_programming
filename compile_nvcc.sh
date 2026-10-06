#!/bin/bash

name="$1"

nvcc -Xcompiler "-O2 -fopenmp" "${name}.cu" -o "$name"
