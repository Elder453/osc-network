#!/bin/bash

# Output CSV file
output_file="experiment_results.csv"

# Create CSV header - renamed first column to model_variant
echo "model_variant,seed,model_type,use_omega,disable_between,symmetric_j,n_train_glyphs,SD_train_acc_final,RMTS_train_acc_final,SD_test_acc,RMTS_test_acc" > "$output_file"

# Process each .out file
for outfile in slogs/*.out; do
    echo "Processing $outfile..."
    
    # Get the base filename (without the extension)
    basename=$(basename "$outfile" .out)
    
    # Find the corresponding .err file
    errfile="slogs/$basename.err"
    
    # Extract parameters section - using a regex that works with the special dash character
    params_section=$(awk '/>>> Hyper.parameters/,/>>> Starting training/' "$outfile")
    
    # Extract parameters using grep and sed
    exp_name=$(echo "$params_section" | grep -E "exp_name[[:space:]]*:" | sed 's/.*:[[:space:]]*\([a-zA-Z0-9_]*\).*/\1/')
    exp_name_prefix=$(echo "$exp_name" | cut -c1-4)
    
    # Map the prefix to descriptive names
    case "$exp_name_prefix" in
        "BTFF") model_variant="baseline" ;;
        "KFFF") model_variant="osc-no-Omega" ;;
        "KFFT") model_variant="osc-symm-J-no-Omega" ;;
        "KTFF") model_variant="osc-default" ;;
        "KTTF") model_variant="osc-no-Jout" ;;
        *) model_variant="$exp_name_prefix" ;; # Keep original if no mapping exists
    esac
    
    seed=$(echo "$params_section" | grep -E "seed[[:space:]]*:" | sed 's/.*:[[:space:]]*\([0-9]*\).*/\1/')
    model_type=$(echo "$params_section" | grep -E "model_type[[:space:]]*:" | sed 's/.*:[[:space:]]*\([a-zA-Z0-9_]*\).*/\1/')
    use_omega=$(echo "$params_section" | grep -E "use_omega[[:space:]]*:" | sed 's/.*:[[:space:]]*\([a-zA-Z]*\).*/\1/')
    disable_between=$(echo "$params_section" | grep -E "disable_between[[:space:]]*:" | sed 's/.*:[[:space:]]*\([a-zA-Z]*\).*/\1/')
    symmetric_j=$(echo "$params_section" | grep -E "symmetric_j[[:space:]]*:" | sed 's/.*:[[:space:]]*\([a-zA-Z]*\).*/\1/')
    n_train_glyphs=$(echo "$params_section" | grep -E "n_train_glyphs[[:space:]]*:" | sed 's/.*:[[:space:]]*\([0-9]*\).*/\1/')
    
    # Extract test accuracies from .out file
    sd_test_acc=$(grep "S/D test loss=" "$outfile" | sed 's/.*acc=\([0-9.]*\).*/\1/')
    rmts_test_acc=$(grep "RMTS test loss=" "$outfile" | sed 's/.*acc=\([0-9.]*\).*/\1/')
    
    # Extract the RMTS final training accuracy from .out file
    rmts_train_acc_final=""
    
    # Extract RMTS section (between "S/D training complete" and "RMTS test loss")
    rmts_section=$(awk '/S\/D training complete/,/RMTS test loss/' "$outfile")
    
    # Get the last "Epoch 50/50" line in the RMTS section
    rmts_epoch_line=$(echo "$rmts_section" | grep "Epoch 50/50")
    
    if [ -n "$rmts_epoch_line" ]; then
        # Extract the train accuracy from the line
        rmts_train_acc_final=$(echo "$rmts_epoch_line" | grep -o "Train Loss=[0-9.]*, Acc=[0-9.]*" | grep -o "Acc=[0-9.]*" | sed 's/Acc=\([0-9.]*\)/\1/')
    fi
    
    # Extract the final SD training accuracy from .err file
    sd_train_acc_final=""
    if [ -f "$errfile" ]; then
        # Find the Epoch 50/50 line and the line immediately after it
        epoch_block=$(grep -A 2 "Epoch 50/50" "$errfile")
        if [ -n "$epoch_block" ]; then
            # Extract the train accuracy from the train loss line
            sd_train_acc_final=$(echo "$epoch_block" | grep "train loss" | grep -o "acc:[0-9.]*" | sed 's/acc:\([0-9.]*\)/\1/')
        fi
    else
        echo "Warning: Corresponding .err file not found: $errfile"
    fi
    
    # Check if all values were extracted successfully
    if [ -z "$model_variant" ] || [ -z "$seed" ] || [ -z "$model_type" ] || [ -z "$use_omega" ] || [ -z "$disable_between" ] || [ -z "$symmetric_j" ] || [ -z "$n_train_glyphs" ] || [ -z "$sd_test_acc" ] || [ -z "$rmts_test_acc" ] || [ -z "$sd_train_acc_final" ] || [ -z "$rmts_train_acc_final" ]; then
        echo "Warning: Could not extract all values from $outfile or $errfile"
        echo "  model_variant: $model_variant (from $exp_name_prefix)"
        echo "  seed: $seed"
        echo "  model_type: $model_type"
        echo "  use_omega: $use_omega"
        echo "  disable_between: $disable_between"
        echo "  symmetric_j: $symmetric_j"
        echo "  n_train_glyphs: $n_train_glyphs"
        echo "  sd_train_acc_final: $sd_train_acc_final"
        echo "  rmts_train_acc_final: $rmts_train_acc_final"
        echo "  sd_test_acc: $sd_test_acc"
        echo "  rmts_test_acc: $rmts_test_acc"
    fi
    
    # Write to CSV using the descriptive model variant name
    echo "$model_variant,$seed,$model_type,$use_omega,$disable_between,$symmetric_j,$n_train_glyphs,$sd_train_acc_final,$rmts_train_acc_final,$sd_test_acc,$rmts_test_acc" >> "$output_file"
done

echo "Results saved to $output_file"