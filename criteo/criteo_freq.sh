for (( i=15; i <= 40; i++ )); do echo $i;  awk -F'\t' -v fnum=$i '{if ($fnum != "") print $fnum}' day_0_small_data.txt | sort | uniq -c | sort -nrk1 | head -n30 | awk -v fnum=$i '{print fnum"\t"$1"\t"$2 }' >> freq30_categorial.txt; done

