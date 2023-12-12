•	Run download_DESIRED.m to download the DESIRED database :
-	In the code, you need to determine the local path to save the database : 
-- path_to_save_records = 'path_to_save_database';
-   Determine the path for the installed WFDB Toolbox depending on your operating system :
*Microsoft: path_to_exes=path_to_WFDB_Toolbox\mcode\nativelibs\windows\bin
* MacOS : 
path_to_exes=path_to_WFDB_Toolbox/mcode/ nativelibs/macosx/bin
* Linux : 
path_to_exes=path_to_WFDB_Toolbox/mcode/nativelibs/linux/bin
		-     For MacOs and Linux users, replace in lines 26 & 28 the following :
			* rdann.exe by rdann
			* wfdb2mat.exe by wfdb2mat
