#!/bin/bash -e

# Create the Brute submission package for homework 5 without config file

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source "$dir/utils.bash"
create_hw_package "05"

