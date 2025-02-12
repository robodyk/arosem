#!/bin/bash -e

# Create the Brute submission package for semestral work

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export ARO_COURSE_ID=1568  # TODO: YEARLY UPDATE
export ARO_BRUTE_SUBMISSION_URL="https://cw.felk.cvut.cz/brute/student/course/${ARO_COURSE_ID}"

function display_file {
	file="$1"
	[ -n "$DISPLAY" ] && dbus-send --session --print-reply --dest=org.freedesktop.FileManager1 \
	  --type=method_call /org/freedesktop/FileManager1 org.freedesktop.FileManager1.ShowItems \
	  array:string:"file://${file}" string:"" > /dev/null
}

destfile="${dir}/sw01.tar.gz"

echo "Creating submission archive for Semestral work:"
(cd "$dir/../.." && tar -cvf "$destfile" -T "${dir}/sw01.files")
echo "Archive created at ${destfile} ."
echo "Submit it to Brute at ${ARO_BRUTE_SUBMISSION_URL}."
  
display_file "${destfile}"

