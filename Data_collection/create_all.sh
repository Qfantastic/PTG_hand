for ((i=1; i<=12; i++))
do
    python3 tryviewer_code.py --folder "20220922_$i"

done


for ((i=1; i<=12; i++))
do
    python3 read_sync_info.py --folder "20220922_$i"
    python3 read_sync_info_left.py --folder "20220922_$i"
done


for ((i=1; i<=12; i++))
do
    python3 read_camerain_info.py --folder "20220922_$i" --save_folder "1022"
done