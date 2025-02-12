dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export ARO_COURSE_ID=1687  # TODO: YEARLY UPDATE
export ARO_BRUTE_SUBMISSION_URL="https://cw.felk.cvut.cz/brute/student/course/${ARO_COURSE_ID}"

function display_file {
	file="$1"
	[ -n "$DISPLAY" ] && dbus-send --session --print-reply --dest=org.freedesktop.FileManager1 \
	  --type=method_call /org/freedesktop/FileManager1 org.freedesktop.FileManager1.ShowItems \
	  array:string:"file://${file}" string:"" > /dev/null
}

function create_hw_package {
	hw="$1"
  if [[ "$hw" == "sw01" ]]; then
    destfile="${dir}/sw01.tar.gz"
  else
    destfile="${dir}/hw${hw}.tar.gz"
  fi
  
  echo "Creating submission archive for ARO homework ${hw}:"
  (cd "$dir/../.." && tar -cvf "$destfile" -T "${dir}/hw${hw}.files")
  echo "Archive created at ${destfile} ."
  echo "Submit it to Brute at ${ARO_BRUTE_SUBMISSION_URL} (don't forget to choose correct homework)."
  
  display_file "${destfile}"
}
