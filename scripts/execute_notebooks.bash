#!/bin/bash

# Exit on error
set -e

# Track failures
FAILED_NOTEBOOKS=()

# Define an array of patterns to skip
SKIP_LIST=("parallel", "experimental")


export PYDEVD_DISABLE_FILE_VALIDATION=1
# Function to print colored text
print_color() {
    case $1 in
        red)
            color_code=$(tput setaf 1)
            ;;
        green)
            color_code=$(tput setaf 2)
            ;;
        yellow)
            color_code=$(tput setaf 3)
            ;;
        blue)
            color_code=$(tput setaf 4)
            ;;
        reset)
            color_code=$(tput sgr0)
            ;;
        *)
            color_code=$(tput sgr0)
            ;;
    esac
    echo -e "${color_code}$2$(tput sgr0)"
}

NOTEBOOKS=$(find ./docs -type f -name "*.ipynb" -not -path '*/.*')

echo $NOTEBOOKS

for file in $NOTEBOOKS
do
    start_time=$(date +%s)  # Start time in seconds

    SKIP_FILE=false
    for pattern in "${SKIP_LIST[@]}"; do
        if [[ "$file" == *"$pattern"* ]]; then
            SKIP_FILE=true
            break
        fi
    done

    if [ "$SKIP_FILE" = true ]; then
        print_color "yellow" "Skipping $file"
        continue
    fi

    print_color "blue" "Executing $file"

    # Execute notebook from its own directory so relative paths work
    notebook_dir=$(dirname "$file")
    notebook_name=$(basename "$file")
    pushd "$notebook_dir" > /dev/null || { print_color "red" "Failed to cd to $notebook_dir"; exit 1; }

    if jupyter nbconvert --to notebook --execute "$notebook_name" --inplace; then
        popd > /dev/null
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        print_color "green" "Execution time for $file: ${elapsed}s"
    else
        popd > /dev/null
        print_color "red" "FAILED: $file"
        FAILED_NOTEBOOKS+=("$file")
    fi
done

# Report failures and exit with error if any
if [ ${#FAILED_NOTEBOOKS[@]} -gt 0 ]; then
    echo ""
    print_color "red" "=== FAILED NOTEBOOKS ==="
    for nb in "${FAILED_NOTEBOOKS[@]}"; do
        print_color "red" "  - $nb"
    done
    exit 1
fi

print_color "green" "All notebooks executed successfully!"
