#!/bin/sh
while read group; do 
    echo "[Ning_$group]"
    echo is_from_file=True
    echo seed_counts=[2]
    echo domain="ning"
    echo graph_folder=./ning_data/
    echo output_file_prefix="ning"
    echo graph_name_prefix=defaultnet.net
    echo heuristics=[KD,GROUPS,SECS]
    echo "executions=range(1)"
    echo goal=None
    echo initialGroups=["'"$group"'"]
    echo
done < ./ning_data/big_groups


