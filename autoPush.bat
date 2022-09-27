set dir=%~dp0
set GIT_PATH="C:/Program Files/Git/mingw64/libexec/git-core/git.exe"

cd %dir%
%GIT_PATH% add --all
%GIT_PATH% commit -m "latest"
%GIT_PATH% push
echo "Push automatically!"