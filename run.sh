nohup python swe_run_mp.py --config config/default.yaml --output_dir ./swe_outputs > mp.txt 2>&1 &
pkill -f "python swe_run_mp.py"