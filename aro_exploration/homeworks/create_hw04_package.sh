#!/bin/bash -e

# Create the Brute submission package for homework 4

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source "$dir/utils.bash"
create_hw_package "04"