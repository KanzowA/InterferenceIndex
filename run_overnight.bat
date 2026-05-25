@echo off
REM ============================================================
REM  Overnight run — GammaLossNN on 2026 MP data
REM  Trains GammaLoss_0 (baseline) and GammaLoss_0.1 on the
REM  102,816-compound hullout_current.json, then scores ξ.
REM
REM  Usage:  run_overnight.bat
REM  Output: mlstabilitytest/ml_data/Ef/allMP_current/
REM          interference_summary_current.csv
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo  Step 1/3: Train GammaLoss_0  (MSE baseline, 2026 data)
echo ============================================================
python mlstabilitytest/train_models.py allMP_current Ef GammaLoss_0
if errorlevel 1 ( echo ERROR in GammaLoss_0 training & exit /b 1 )

echo.
echo ============================================================
echo  Step 2/3: Train GammaLoss_0.1  (iiLoss, 2026 data)
echo ============================================================
python mlstabilitytest/train_models.py allMP_current Ef GammaLoss_0.1
if errorlevel 1 ( echo ERROR in GammaLoss_0.1 training & exit /b 1 )

echo.
echo ============================================================
echo  Step 3/3: Compute interference scores
echo ============================================================
python interference_score.py allMP_current --hullout hullout_current.json
if errorlevel 1 ( echo ERROR in interference_score.py & exit /b 1 )

echo.
echo ============================================================
echo  All done. Results in interference_summary_current.csv
echo ============================================================
