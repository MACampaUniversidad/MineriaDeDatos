echo "------------ Python 2.7 ------------"
export SAVRW_USE_CWRITEROW="off" 
python profile.py
export SAVRW_USE_CWRITEROW="on" 
python profile.py

echo "------------ Python 3.3 ------------"
export SAVRW_USE_CWRITEROW="off" 
python profile.py
export SAVRW_USE_CWRITEROW="on" 
python profile.py

