for ((i=1; i<=6; i++))
do
    python3 read_sync_info.py --folder "20221007_5-$i"
    python3 read_sync_info_left.py --folder "20221007_5-$i"
done