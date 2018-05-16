python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_ideal_P10_BG_logcal/" -MP -pointJy 10
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_ideal_P100_BG_logcal/" -MP -pointJy 100
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_ideal_P1000_BG_logcal/" -MP -pointJy 1000

python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_ideal_P10_BG_fullcal/" -MP -pointJy 10 -cal_mode "full"
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_ideal_P100_BG_fullcal/" -MP -pointJy 100 -cal_mode "full"
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_ideal_P1000_BG_fullcal/" -MP -pointJy 1000 -cal_mode "full"


python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_offset_P10_BG_logcal/" -MP -pointJy 10 -offset
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_offset_P100_BG_logcal/" -MP -pointJy 100 -offset
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_offset_P1000_BG_logcal/" -MP -pointJy 1000 -offset

python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_offset_P10_BG_fullcal/" -MP -pointJy 10 -cal_mode "full" -offset
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_offset_P100_BG_fullcal/" -MP -pointJy 100 -cal_mode "full" -offset
python CRAMPS.py -path "/home/rjoseph/Bulk/Redundant_Calibration/Simulation_Output/CRAMPS_Linear_offset_P1000_BG_fullcal/" -MP -pointJy 1000 -cal_mode "full" -offset




